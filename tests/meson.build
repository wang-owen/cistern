fixture_args = ['-DCISTERN_FIXTURE_DIR="' + meson.current_source_dir() + '"']

unit_exe = executable('cistern-unit',
  files('tensor_table_test.cc', 'align_test.cc', 'state_test.cc',
        'crc32c_test.cc', 'layout_test.cc', 'writer_test.cc',
        'restore_test.cc'),
  cpp_args: fixture_args, dependencies: [cistern_dep, catch2])
test('unit', unit_exe, suite: 'unit', timeout: 120)

# Separate executable: crash_hooks.cc interposes libc and liburing calls.
crash_exe = executable('cistern-crash-test',
  files('crash_test.cc', 'crash_hooks.cc'),
  cpp_args: fixture_args,
  # std::filesystem::rename calls rename() from inside libstdc++.so
  link_args: cxx.get_supported_link_arguments('-Wl,--export-dynamic-symbol=rename'),
  dependencies: [cistern_dep, catch2, cxx.find_library('dl', required: false)])
test('crash', crash_exe, suite: 'crash', timeout: 300)
