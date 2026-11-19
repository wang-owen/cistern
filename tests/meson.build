fixture_args = ['-DCISTERN_FIXTURE_DIR="' + meson.current_source_dir() + '/fixtures"']

unit_exe = executable('cistern-unit',
  files('test_tensor_table.cpp', 'test_align.cpp', 'test_state.cpp', 'test_env.cpp',
        'test_baselines.cpp'),
  cpp_args: fixture_args, dependencies: [cistern_dep, catch2])
test('unit', unit_exe, suite: 'unit', timeout: 120)

if get_option('owner_impl')
  seam_exe = executable('cistern-seam',
    files('test_roundtrip.cpp', 'test_crash.cpp'),
    cpp_args: fixture_args, dependencies: [cistern_dep, catch2])
  test('roundtrip', seam_exe, args: ['[seam][roundtrip]'], suite: 'seam', timeout: 300)
  test('crash', seam_exe, args: ['[seam][crash]'], suite: 'seam', timeout: 3600)
endif
