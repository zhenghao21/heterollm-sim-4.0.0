#pragma once
#include "../../src/llama-ext.h"

// Host boundary tracing is opt-in at compile time. Define LLAMA_SERVER_HOST_TRACE
// for developer builds; production binaries keep the calls compiled out.
#if defined(LLAMA_SERVER_HOST_TRACE)
#define SERVER_HOST_MARK(label) llama_trace_mark(label)
#define SERVER_HOST_RANGE_START(label) llama_trace_range_start(label)
#define SERVER_HOST_RANGE_END(id) llama_trace_range_end(id)
#else
#define SERVER_HOST_MARK(label) do { (void) sizeof(label); } while (0)
#define SERVER_HOST_RANGE_START(label) (0ULL)
#define SERVER_HOST_RANGE_END(id) do { (void) (id); } while (0)
#endif

