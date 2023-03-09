envoi mpi des configurations:
vortices.container (std::vector<double> de taille variable), vortices.vector_x, vortices.vector_y,
int isMobile,
grid.width, grid.height (size_t), grid.m_left, grid.bottom, grid.step (double), grid.velocity_field (std::vector<(double, double)>),
cloudOfPoints.setOfPoints (std::vector<(double, double)>)

Est-ce qu'il faut faire MPI_Type_free pour les types crées ? Ou est-ce que ca se free automatiquement ?

Pour la suite, qd multithreading:
MPI_Init -> MPI_Init_thread avec MPI_THREAD_MULTIPLE
