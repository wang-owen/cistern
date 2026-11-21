project('cistern', 'cpp',
  version: '0.1.0',
  meson_version: '>=1.3',
  default_options: [
    'cpp_std=c++23',
    'warning_level=3',
    'werror=true',
    'force_fallback_for=catch2,nlohmann_json',
  ])

cxx = meson.get_compiler('cpp')

# Applies to this project only; subprojects keep their own flags (see below).
add_project_arguments(
  cxx.get_supported_arguments(
    '-Wshadow', '-Wconversion', '-Wnon-virtual-dtor', '-Wold-style-cast',
    '-Wdouble-promotion'),
  language: 'cpp')
# GCC < 14 raises false positives inside libstdc++ streambuf at -O3 (13.3, via TensorTable::load).
if cxx.get_id() != 'gcc' or cxx.version().version_compare('>=14')
  add_project_arguments(cxx.get_supported_arguments('-Wnull-dereference'), language: 'cpp')
endif
if get_option('native')
  add_project_arguments('-march=native', language: 'cpp')
endif
add_project_arguments('-DCISTERN_SOURCE_DIR="' + meson.project_source_root() + '"', language: 'cpp')

# --- environment checks at configure time (warnings only; the binary hard-checks at startup) ---
kr = run_command('sh', '-c', 'uname -r | sed "s/[^0-9.].*//"', check: false)
if kr.returncode() == 0 and kr.stdout().strip() != '' and kr.stdout().strip().version_compare('<6.1')
  warning('build host kernel ' + kr.stdout().strip() + ' is older than 6.1; the startup checks refuse to run on it')
endif

liburing = dependency('liburing', version: '>=2.5')

sub_opts = ['werror=false', 'warning_level=0', 'default_library=static']
catch2 = dependency('catch2-with-main',
  fallback: ['catch2', 'catch2_with_main_dep'],
  default_options: sub_opts + ['tests=false'], include_type: 'system')
json = dependency('nlohmann_json',
  fallback: ['nlohmann_json', 'nlohmann_json_dep'],
  default_options: sub_opts, include_type: 'system')

owner_impl = get_option('owner_impl')

lib_sources = files(
  'src/align.cpp', 'src/tensor_table.cpp', 'src/state.cpp', 'src/env.cpp', 'src/baselines.cpp',
  # owner stubs: compiled so the seam headers stay valid
  'src/writer.cpp', 'src/reader.cpp', 'src/async.cpp', 'src/adam.cpp')
inc = include_directories('include')
libcistern = static_library('cistern', lib_sources,
  include_directories: inc, dependencies: [liburing, json], gnu_symbol_visibility: 'hidden')
cistern_dep = declare_dependency(link_with: libcistern, include_directories: inc,
  dependencies: [liburing, json])

subdir('tests')

executable('cistern-baseline', files('bench/baseline_cli.cpp'), dependencies: [cistern_dep])
