/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A CUDA Graph whose IF conditional node is decided by a kernel.
//!
//! ```text
//! select(handle, flag) --> IF(handle) { body(out) }
//! ```
//!
//! `select` reads a flag from device memory and sets the conditional with
//! `cuda_device::graph::set_conditional`; the body writes 42 only when the
//! flag is nonzero. The same instantiated graph is launched with the flag set
//! and cleared, so the branch is taken on the device, not by rebuilding the
//! graph.

use cuda_core::sys;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::graph::{ConditionalHandle, set_conditional};
use cuda_device::{cuda_module, kernel, thread};
use std::ffi::c_void;
use std::ptr::{null, null_mut};

#[cuda_module]
mod kernels {
    use super::*;

    /// Set the conditional node to `*flag`.
    ///
    /// # Safety
    ///
    /// `handle` belongs to the launching graph and precedes its node; `flag`
    /// points to one readable `u32`.
    #[kernel]
    pub unsafe fn select(handle: ConditionalHandle, flag: *const u32) {
        if thread::threadIdx_x() == 0 {
            unsafe { set_conditional(handle, *flag) };
        }
    }

    /// The conditional body.
    ///
    /// # Safety
    ///
    /// `out` points to one writable `u32`.
    #[kernel]
    pub unsafe fn body(out: *mut u32) {
        if thread::threadIdx_x() == 0 {
            unsafe { *out = 42 };
        }
    }
}

fn check(result: sys::CUresult, call: &str) {
    assert_eq!(
        result,
        sys::cudaError_enum_CUDA_SUCCESS,
        "{call} failed with CUresult {result}"
    );
}

/// One-block kernel node parameters for `function` with `args`.
fn kernel_node(
    function: sys::CUfunction,
    args: &mut [*mut c_void],
) -> sys::CUDA_KERNEL_NODE_PARAMS {
    // SAFETY: an all-zero CUDA_KERNEL_NODE_PARAMS is valid; the fields a
    // launch needs are set below.
    let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
    params.func = function;
    (params.gridDimX, params.gridDimY, params.gridDimZ) = (1, 1, 1);
    (params.blockDimX, params.blockDimY, params.blockDimZ) = (32, 1, 1);
    params.kernelParams = args.as_mut_ptr();
    params
}

fn main() {
    let ctx = CudaContext::new(0).expect("create a CUDA context");
    let stream = ctx.default_stream();
    let module = cuda_host::load_embedded_module(&ctx, env!("CARGO_PKG_NAME"))
        .expect("load the embedded module");
    let select = module.load_function("select").expect("find select");
    let body = module.load_function("body").expect("find body");

    let mut flag = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let flag_ptr = flag.cu_deviceptr();
    let out_ptr = out.cu_deviceptr();

    // SAFETY: every call below follows the CUDA driver API contract; the
    // argument arrays and the values they point to outlive node creation.
    let exec = unsafe {
        let mut graph: sys::CUgraph = null_mut();
        check(sys::cuGraphCreate(&mut graph, 0), "cuGraphCreate");
        let mut raw_handle: sys::CUgraphConditionalHandle = 0;
        check(
            sys::cuGraphConditionalHandleCreate(
                &mut raw_handle,
                graph,
                ctx.cu_ctx(),
                0,
                sys::CU_GRAPH_COND_ASSIGN_DEFAULT,
            ),
            "cuGraphConditionalHandleCreate",
        );
        let handle = ConditionalHandle::from_raw(raw_handle);

        let mut select_args = [
            (&raw const handle).cast_mut().cast::<c_void>(),
            (&raw const flag_ptr).cast_mut().cast::<c_void>(),
        ];
        let select_params = kernel_node(select.cu_function(), &mut select_args);
        let mut select_node: sys::CUgraphNode = null_mut();
        check(
            sys::cuGraphAddKernelNode_v2(&mut select_node, graph, null(), 0, &select_params),
            "cuGraphAddKernelNode (select)",
        );

        let mut conditional: sys::CUgraphNodeParams = std::mem::zeroed();
        conditional.type_ = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_CONDITIONAL;
        conditional.__bindgen_anon_1.conditional = sys::CUDA_CONDITIONAL_NODE_PARAMS {
            handle: raw_handle,
            type_: sys::CUgraphConditionalNodeType_enum_CU_GRAPH_COND_TYPE_IF,
            size: 1,
            phGraph_out: null_mut(),
            ctx: ctx.cu_ctx(),
        };
        let mut conditional_node: sys::CUgraphNode = null_mut();
        check(
            sys::cuGraphAddNode_v2(
                &mut conditional_node,
                graph,
                &select_node,
                null(),
                1,
                &mut conditional,
            ),
            "cuGraphAddNode (conditional)",
        );
        let body_graph = *conditional.__bindgen_anon_1.conditional.phGraph_out;

        let mut body_args = [(&raw const out_ptr).cast_mut().cast::<c_void>()];
        let body_params = kernel_node(body.cu_function(), &mut body_args);
        let mut body_node: sys::CUgraphNode = null_mut();
        check(
            sys::cuGraphAddKernelNode_v2(&mut body_node, body_graph, null(), 0, &body_params),
            "cuGraphAddKernelNode (body)",
        );

        let mut exec: sys::CUgraphExec = null_mut();
        check(
            sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
            "cuGraphInstantiate",
        );
        check(sys::cuGraphDestroy(graph), "cuGraphDestroy");
        exec
    };

    let mut failures = 0;
    for (flag_value, expected) in [(1_u32, 42_u32), (0, 0), (7, 42)] {
        flag.copy_from_host(&stream, &[flag_value]).unwrap();
        out.copy_from_host(&stream, &[0]).unwrap();
        // SAFETY: `exec` is a live executable graph of this context.
        unsafe {
            check(
                sys::cuGraphLaunch(exec, stream.cu_stream()),
                "cuGraphLaunch",
            )
        };
        let got = out.to_host_vec(&stream).unwrap()[0];
        let verdict = if got == expected { "ok" } else { "WRONG" };
        println!("flag {flag_value}: body wrote {got} (expected {expected}) {verdict}");
        failures += usize::from(got != expected);
    }
    // SAFETY: the graph is idle after the last synchronizing copy.
    unsafe { check(sys::cuGraphExecDestroy(exec), "cuGraphExecDestroy") };

    if failures == 0 {
        println!("\n✓ SUCCESS: the device selected the conditional branch");
    } else {
        println!("\n✗ FAILED: {failures} launch(es) took the wrong branch");
        std::process::exit(1);
    }
}
