Pour la suite, qd multithreading:
MPI_Init -> MPI_Init_thread avec MPI_THREAD_MULTIPLE

Commande de lancement:
clear && make all && mpiexec -n 1 ./vortexSimulation.exe data/simpleSimulation.dat 1280 1024

Problème de la méthode précédente: les structures contiennent des pointeurs et pas les données elles même
Il faut donc envoyer les données "à la main"
