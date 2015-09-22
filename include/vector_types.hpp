//
// Created by pfarre on 22/09/15.
//

#pragma once

#include <cuda_utils.hpp>
#include <vector_type_traits.hpp>


template <typename T>
struct vec3 {
    using Type     = vec3<T>;
    using VecType  = typename vector_type_traits<Type>::VecType;
    using BaseType = T;

    T x;
    T y;
    T z;

    HOST_DEVICE_INLINE vec3()
    {}

    HOST_DEVICE_INLINE vec3(VecType v)
            : x{v.x}, y{v.y}, z{v.z}
    {}

    HOST_DEVICE_INLINE vec3(BaseType x, BaseType y, BaseType z)
            : x{x}, y{y}, z{z}
    {}

    HOST_DEVICE_INLINE Type& operator+=(const Type other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator-=(const Type other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(const Type other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(BaseType scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    HOST_DEVICE_INLINE BaseType dot() const {
        return x*x + y*y + z*z;
    }
};

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator+(vec3<T> rhs, vec3<T> lhs) {
    return rhs += lhs;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator-(vec3<T> rhs, vec3<T> lhs) {
    return rhs -= lhs;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(vec3<T> rhs, vec3<T> lhs) {
    return rhs *= lhs;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(vec3<T> rhs, T scalar) {
    return rhs *= scalar;
}

template <typename T>
HOST_DEVICE_INLINE vec3<T> operator*(T scalar, vec3<T> lhs) {
    return lhs *= scalar;
}

template <typename T>
struct vec4 {
    using Type     = vec4<T>;
    using VecType  = typename vector_type_traits<Type>::VecType;
    using BaseType = T;

    T x; // x component
    T y; // y component
    T z; // z component
    T w; // mass

    HOST_DEVICE_INLINE vec4()
    {}

    HOST_DEVICE_INLINE vec4(VecType v)
            : x{v.x}, y{v.y}, z{v.z}, w{v.w}
    {}

    HOST_DEVICE_INLINE vec4(BaseType x, BaseType y, BaseType z, BaseType w)
            : x{x}, y{y}, z{z}, w{w}
    {}

    HOST_DEVICE_INLINE explicit operator VecType() const {
        return {x, y, z, w};
    }

    HOST_DEVICE_INLINE Type& operator+=(const Type other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator-=(const Type other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(const Type other) {
        x *= other.x;
        y *= other.y;
        z *= other.z;
        w *= other.w;
        return *this;
    }

    HOST_DEVICE_INLINE Type& operator*=(BaseType scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

};

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator*(vec4<T> rhs, T scalar) {
    return rhs *= scalar;
}

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator*(T scalar, vec4<T> lhs) {
    return lhs *= scalar;
}

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator+(vec4<T> rhs, vec4<T> lhs) {
    vec4<T> ret;
    ret.x = rhs.x + lhs.x;
    ret.y = rhs.y + lhs.y;
    ret.z = rhs.z + lhs.z;
    ret.w = rhs.w + lhs.w;
    return ret;
}

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator-(vec4<T> rhs, vec4<T> lhs) {
    vec4<T> ret;
    ret.x = rhs.x - lhs.x;
    ret.y = rhs.y - lhs.y;
    ret.z = rhs.z - lhs.z;
    ret.w = rhs.w - lhs.w;
    return ret;
}

template <typename T>
HOST_DEVICE_INLINE vec4<T> operator*(vec4<T> rhs, vec4<T> lhs) {
    vec4<T> ret;
    ret.x = rhs.x * lhs.x;
    ret.y = rhs.y * lhs.y;
    ret.z = rhs.z * lhs.z;
    ret.w = rhs.w * lhs.w;
    return ret;
}