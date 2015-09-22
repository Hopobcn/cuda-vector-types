//
// Created by pfarre on 22/09/15.
//

#pragma once

#include <vector_types.hpp>

template <typename T>
struct vec3;
template <typename T>
struct vec4;

template <typename T>
struct vector_type_traits {
    using VecType = T;
};

template <>
struct vector_type_traits<vec3<float>> {
    using VecType = float3;
};

template <>
struct vector_type_traits<vec3<double>> {
    using VecType = double3;
};

template <>
struct vector_type_traits<vec4<float>> {
    using VecType = float4;
};

template <>
struct vector_type_traits<vec4<double>> {
    using VecType = double4;
};