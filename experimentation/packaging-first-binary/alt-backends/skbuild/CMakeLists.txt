cmake_minimum_required(VERSION 3.15)
project(d2shim_skb C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)   # MSVC: export sm_* without __declspec / .def
add_library(nashim SHARED src/d2shim_skb/c/nashim.c src/d2shim_skb/vendor/nanoarrow.c)
target_include_directories(nashim PRIVATE src/d2shim_skb/vendor)
install(TARGETS nashim LIBRARY DESTINATION d2shim_skb RUNTIME DESTINATION d2shim_skb)
