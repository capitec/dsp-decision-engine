project('d2shim_mp', 'c', version: '0.0.1', default_options: ['c_std=c99', 'buildtype=release'])
py = import('python').find_installation()
pkg = 'd2shim_mp'
shared_library('nashim',
  'src/d2shim_mp/c/nashim.c', 'src/d2shim_mp/vendor/nanoarrow.c',
  include_directories: 'src/d2shim_mp/vendor',
  install: true, install_dir: py.get_install_dir(pure: false) / pkg,   # pure: false, or the wheel is tagged py3-none-ANY with an x86_64 .so inside (observed)
  name_prefix: 'lib')
py.install_sources('src/d2shim_mp/__init__.py', subdir: pkg, pure: false)
