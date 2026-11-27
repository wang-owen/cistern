project('pitstop', 'cpp',
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
add_project_arguments('-DPITSTOP_SOURCE_DIR="' + meson.project_source_root() + '"', language: 'cpp')

liburing = dependency('liburing', version: '>=2.5')

sub_opts = ['werror=false', 'warning_level=0', 'default_library=static']
catch2 = dependency('catch2-with-main',
  fallback: ['catch2', 'catch2_with_main_dep'],
  default_options: sub_opts + ['tests=false'], include_type: 'system')
json = dependency('nlohmann_json',
  fallback: ['nlohmann_json', 'nlohmann_json_dep'],
  default_options: sub_opts, include_type: 'system')

subdir('src')
subdir('tests')
subdir('tools')
