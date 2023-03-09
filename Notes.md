envoi mpi des configurations:
vortices.container (std::vector<double> de taille variable), vortices.vector_x, vortices.vector_y,
int isMobile,
grid.width, grid.height (size_t), grid.m_left, grid.bottom, grid.step (double), grid.velocity_field (std::vector<(double, double)>),
cloudOfPoints.setOfPoints (std::vector<(double, double)>)

Envoi de messages asynchrones sur globComm pour fin de simulation...

Pour la suite, qd multithreading:
MPI_Init -> MPI_Init_thread avec MPI_THREAD_MULTIPLE
