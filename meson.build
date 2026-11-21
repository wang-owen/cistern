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

# --- HIP (optional; M1 does not need it) ---
hip_opt = get_option('hip')
hip_dep = dependency('', required: false)  # not-found placeholder
hip_gen = disabler()
if hip_opt.allowed()
  hipcc = find_program('hipcc', required: hip_opt)
  hipconfig = find_program('hipconfig', required: false)
  if hipcc.found()
    hip_root = '/usr'
    if hipconfig.found()
      r = run_command(hipconfig, '--path', check: false)
      if r.returncode() == 0 and r.stdout().strip() != ''
        hip_root = r.stdout().strip()
      endif
    endif
    hip_libdirs = [hip_root / 'lib', hip_root / 'lib64', '/usr/lib/x86_64-linux-gnu']
    hip_lib = cxx.find_library('amdhip64', dirs: hip_libdirs, required: hip_opt)
    hip_inc = include_directories(hip_root / 'include', is_system: true)
    if hip_lib.found() and cxx.has_header('hip/hip_runtime_api.h',
        args: ['-D__HIP_PLATFORM_AMD__', '-isystem' + hip_root / 'include'],
        required: hip_opt)
      hip_dep = declare_dependency(
        dependencies: hip_lib,
        include_directories: hip_inc,
        compile_args: ['-D__HIP_PLATFORM_AMD__', '-DCISTERN_HAVE_HIP=1'])
      # hipcc compiles .hip -> .o; link the objects into the target that uses them.
      hip_gen = generator(hipcc,
        output: '@BASENAME@.o',
        arguments: ['--offload-arch=' + get_option('hip_arch'), '-O3', '-fPIC', '-std=c++20',
                    '-c', '@INPUT@', '-o', '@OUTPUT@'])
      message('HIP enabled: root=' + hip_root + ' arch=' + get_option('hip_arch'))
    endif
  endif
endif

subdir('src')
subdir('bench/baselines')
subdir('tests')
