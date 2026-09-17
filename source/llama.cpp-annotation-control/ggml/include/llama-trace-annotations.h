#pragma once

#include <cstdlib>

// Freeze the process environment choice on first use in each translation unit.
// Set LLAMA_TRACE_ANNOTATIONS=1 before startup to keep semantic annotations.
static inline bool llama_trace_annotations_enabled() {
    static const bool enabled = []() {
        const char * value = std::getenv("LLAMA_TRACE_ANNOTATIONS");
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
