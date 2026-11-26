# libcistern: shared pieces (common, table, gen) plus, later, the owner's layout/writer/restore.
inc = include_directories('.')

common_sources = files('common/align.cc', 'common/env.cc')
table_sources = files('table/tensor_table.cc')
gen_sources = files('gen/state.cc')

# Owner code goes here as it is written, e.g.
#   'layout/layout.cc', 'writer/writer.cc', 'restore/restore.cc'
owner_sources = files('layout/layout.cc', 'writer/writer.cc')

libcistern = static_library('cistern',
  common_sources, table_sources, gen_sources, owner_sources,
  include_directories: inc,
  dependencies: [liburing, json],
  gnu_symbol_visibility: 'hidden')

cistern_dep = declare_dependency(
  link_with: libcistern,
  include_directories: inc,
  dependencies: [liburing, json])
