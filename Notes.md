Ecrire les /_ A FAIRE ! _/

Envoi de messages asynchrones sur globComm pour fin de simulation ?

Pour la suite, qd multithreading:
MPI_Init -> MPI_Init_thread avec MPI_THREAD_MULTIPLE

On multiplie les datatype pour éviter d'envoyer des données inutiles (ex SimulationDataMobile et SendConfig)
