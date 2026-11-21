fixture_args = ['-DCISTERN_FIXTURE_DIR="' + meson.current_source_dir() + '/fixtures"']

unit_exe = executable('cistern-unit',
  files('test_smoke.cc', 'test_tensor_table.cc', 'test_align.cc', 'test_state.cc',
        'test_env.cc', 'test_baselines.cc'),
  cpp_args: fixture_args, dependencies: [cistern_dep, baselines_dep, catch2])
test('unit', unit_exe, suite: 'unit', timeout: 120)

py = find_program('python3', required: false)
if py.found()
  test('tensor-table-py', py,
    args: ['-I', files('python/test_tensor_table.py')],
    suite: 'python', timeout: 60)
endif

if hip_dep.found()
  hip_exe = executable('cistern-hip',
    [files('hip/test_hip.cc'), hip_gen.process('hip/smoke_kernel.hip')],
    dependencies: [hip_dep, catch2])
  test('hip', hip_exe, suite: 'hip', timeout: 60)
endif
