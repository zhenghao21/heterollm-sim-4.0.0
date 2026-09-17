#pragma once
#include "launch_decode.h"
#include <iomanip>
#include <sstream>

namespace mmvq_capture {
inline std::string opaque_attribute_hex(const CapturedAttribute & a) {
    std::ostringstream s;s<<std::hex<<std::setfill('0');
    for(auto value:a.opaque_value)s<<std::setw(2)<<unsigned(value);
    return s.str();
}
inline std::string launch_metadata_json(const Launch & r) {
    std::ostringstream s;
    const auto boolean=[](bool b){return b?"true":"false";};
    const auto vector=[](dim3 v){return '['+std::to_string(v.x)+','+std::to_string(v.y)+','+std::to_string(v.z)+']';};
    s<<"\"geometry_observed\":"<<boolean(r.geometry_copied)
     <<",\"function\":"<<(r.geometry_copied?std::to_string(r.function):"null")
     <<",\"stream\":"<<(r.geometry_copied?std::to_string(r.stream):"null")
     <<",\"grid\":"<<(r.geometry_copied?vector(r.grid):"null")
     <<",\"block\":"<<(r.geometry_copied?vector(r.block):"null")
     <<",\"shared\":"<<(r.geometry_copied?std::to_string(r.shared):"null")
     <<",\"extended_api\":"<<boolean(r.extended_api)
     <<",\"api_name_mismatch\":"<<boolean(r.api_name_mismatch)
     <<",\"missing_config\":"<<boolean(r.missing_config)
     <<",\"attribute_list_observed\":"<<boolean(r.extended_api&&r.geometry_copied&&!r.missing_attributes)
     <<",\"missing_attributes\":"<<boolean(r.missing_attributes)
     <<",\"truncated_attributes\":"<<boolean(r.truncated_attributes)
     <<",\"unknown_attributes\":"<<boolean(r.unknown_attributes)
     <<",\"attributes_source_qualified\":"<<boolean(r.attributes_source_qualified)
     <<",\"reported_attribute_count\":"<<(r.extended_api&&r.geometry_copied?std::to_string(r.reported_attribute_count):"null")
     <<",\"captured_attribute_count\":"<<(r.extended_api&&r.geometry_copied?std::to_string(r.captured_attribute_count):"null")
     <<",\"attributes\":";
    if(!r.extended_api||!r.geometry_copied||r.missing_attributes){s<<"null";return s.str();}
    s<<'[';
    for(unsigned i=0;i<r.captured_attribute_count;++i){
        if(i)s<<',';const auto &a=r.attributes[i];
        s<<"{\"index\":"<<i<<",\"id\":"<<a.id<<",\"is_source_PDL\":"<<boolean(a.is_pdl)
         <<",\"programmaticStreamSerializationAllowed\":"<<(a.is_pdl?std::to_string(a.programmatic_serialization_allowed):"null")
         <<",\"opaque_value_bytes_hex\":\""<<opaque_attribute_hex(a)<<"\""
         <<",\"interpreted_prefix_bytes\":"<<(a.is_pdl?4:0)
         <<",\"inactive_union_bytes_are_semantic\":false}";
    }
    s<<']';return s.str();
}
} // namespace mmvq_capture
