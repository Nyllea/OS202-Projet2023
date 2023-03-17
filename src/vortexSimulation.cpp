#include <SFML/Window/Keyboard.hpp>
#include <ios>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include "cartesian_grid_of_speed.hpp"
#include "vortex.hpp"
#include "cloud_of_points.hpp"
#include "runge_kutta.hpp"
#include "screen.hpp"

#include <mpi.h>
#include "vector.hpp"

/* --------------------------------------Déclaration des structures nécessaires à la communication entre processus ----------------------------------------*/

// Structure pour envoyer une configuration entière
struct FullConfig
{
    Simulation::Vortices vortices;
    int isMobile;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloud;
};

struct SimulationCommands
{
    bool endSimulation;
    bool animate;
    bool advance;
    double dt;
};

struct InitializationData
{
    bool dataLoaded;

    int modeGeneration;
    int isMobile;

    std::size_t nx, ny;
    std::size_t nbPoints;
    std::size_t nbVortices;

    double xleft, ybot, h;
    double xl, yb, xr, yt;

    double *x, *y, *force;
};

/* --------------------------------------------- Fonctions de création/suppression des Datatype MPI nécessaires -------------------------------------------- */

template <typename elementType>
MPI_Datatype create_vector_datatype(MPI_Datatype mpi_elementType, int vectorLength)
{
    MPI_Datatype vectorType, doubleType;

    MPI_Type_contiguous(vectorLength, mpi_elementType, &doubleType);
    MPI_Type_create_resized(doubleType, 0, vectorLength * sizeof(elementType), &vectorType);

    MPI_Type_commit(&vectorType);
    MPI_Type_free(&doubleType);

    return vectorType;
}

template <typename RealType>
MPI_Datatype create_geometry_vector_datatype()
{
    int const count = 2;

    int blocklengths[count] = {1, 1};
    MPI_Datatype types[count] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[count] = {offsetof(Geometry::Vector<RealType>, x),
                               offsetof(Geometry::Vector<RealType>, y)};

    MPI_Datatype geomVect_DT;

    MPI_Type_create_struct(count, blocklengths, offsets, types, &geomVect_DT);
    MPI_Type_create_resized(geomVect_DT, 0, sizeof(Geometry::Vector<RealType>), &geomVect_DT);
    MPI_Type_commit(&geomVect_DT);

    return geomVect_DT;
}

template <typename RealType>
MPI_Datatype create_point_datatype(MPI_Datatype mpi_realType_DT)
{
    int count = 2;

    int block_lengths[count] = {1, 1};
    MPI_Aint displacements[count] = {offsetof(Geometry::Point<RealType>, x), offsetof(Geometry::Point<RealType>, y)};
    MPI_Datatype types[count] = {mpi_realType_DT, mpi_realType_DT};

    MPI_Datatype point_DT;
    MPI_Type_create_struct(count, block_lengths, displacements, types, &point_DT);
    MPI_Type_create_resized(point_DT, 0, sizeof(Geometry::Point<RealType>), &point_DT);
    MPI_Type_commit(&point_DT);

    return point_DT;
}

MPI_Datatype create_simulationCommands_datatype()
{
    int count = 4;

    int block_lengths[count] = {1, 1, 1, 1};

    MPI_Aint displacements[count] = {
        offsetof(SimulationCommands, endSimulation),
        offsetof(SimulationCommands, animate),
        offsetof(SimulationCommands, advance),
        offsetof(SimulationCommands, dt)};

    MPI_Datatype types[count] = {MPI_CXX_BOOL, MPI_CXX_BOOL, MPI_CXX_BOOL, MPI_DOUBLE};

    MPI_Datatype simulationCommands_DT;
    MPI_Type_create_struct(count, block_lengths, displacements, types, &simulationCommands_DT);
    MPI_Type_create_resized(simulationCommands_DT, 0, sizeof(SimulationCommands), &simulationCommands_DT);
    MPI_Type_commit(&simulationCommands_DT);

    return simulationCommands_DT;
}

MPI_Datatype create_initializationData_datatype()
{
    int count = 5;

    int block_lengths[count] = {1, 2, 4, 7, 3};

    MPI_Aint displacements[count] = {
        offsetof(InitializationData, dataLoaded),
        offsetof(InitializationData, modeGeneration),
        offsetof(InitializationData, nx),
        offsetof(InitializationData, xleft),
        offsetof(InitializationData, x)};

    MPI_Datatype types[count] = {MPI_CXX_BOOL, MPI_INT, MPI_UNSIGNED_LONG, MPI_DOUBLE, MPI_DOUBLE};

    MPI_Datatype initData_DT;
    MPI_Type_create_struct(count, block_lengths, displacements, types, &initData_DT);
    MPI_Type_create_resized(initData_DT, 0, sizeof(InitializationData), &initData_DT);
    MPI_Type_commit(&initData_DT);

    return initData_DT;
}

// Nettoie toutes les variables MPI et appelle MPI_Finalize()
void StopMPI(
    MPI_Datatype *simulationCommands_DT, MPI_Datatype *initializationData_DT,
    MPI_Datatype *cloudPoint_DT,
    MPI_Datatype *geomVect_DT)
{
    MPI_Type_free(initializationData_DT);

    MPI_Type_free(simulationCommands_DT);

    MPI_Type_free(cloudPoint_DT);

    MPI_Type_free(geomVect_DT);

    MPI_Finalize();
}

/* ------------------------------------------ Fonctions de récupération des données initiale de la configuration ---------------------------------- */
// Retourne toutes les informations nécessaires à l'initialisation des classes à partir du fichier
InitializationData getConfig(std::ifstream &input)
{
    InitializationData initData;

    constexpr std::size_t maxBuffer = 8192;
    char buffer[maxBuffer];
    std::string sbuffer;
    std::stringstream ibuffer;
    // Lit la première ligne de commentaire :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lecture de la grille cartésienne
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> initData.xleft >> initData.ybot >> initData.nx >> initData.ny >> initData.h;
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit mode de génération des particules
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> initData.modeGeneration;
    if (initData.modeGeneration == 0) // Génération sur toute la grille
    {
        ibuffer >> initData.nbPoints;
    }
    else
    {
        ibuffer >> initData.xl >> initData.yb >> initData.xr >> initData.yt >> initData.nbPoints;
    }
    // Lit le nombre de vortex :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le nombre de vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    try
    {
        ibuffer >> initData.nbVortices;
    }
    catch (std::ios_base::failure &err)
    {
        std::cout << "Error " << err.what() << " found" << std::endl;
        std::cout << "Read line : " << sbuffer << std::endl;
        throw err;
    }
    input.getline(buffer, maxBuffer); // Relit un commentaire
    initData.x = new double[initData.nbVortices];
    initData.y = new double[initData.nbVortices];
    initData.force = new double[initData.nbVortices];
    for (std::size_t iVortex = 0; iVortex < initData.nbVortices; ++iVortex)
    {
        input.getline(buffer, maxBuffer);
        std::string sbuffer(buffer, maxBuffer);
        std::stringstream ibuffer(sbuffer);
        ibuffer >> initData.x[iVortex] >> initData.y[iVortex] >> initData.force[iVortex];
    }
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le mode de déplacement des vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> initData.isMobile;

    return initData;
}

// Retourne les classes initialisées à partir des informations d'initialisation dans initData
void readConfig(InitializationData &initData, const int rank, const int nbpCalc, FullConfig &fullConfig)
{
    using point = Simulation::Vortices::point;

    fullConfig.cartesianGrid = Numeric::CartesianGridOfSpeed({initData.nx, initData.ny}, point{initData.xleft, initData.ybot}, initData.h);

    int x_local = 0;
    int nbPoints_local = initData.nbPoints;

    if (rank != 0)
    {
        std::size_t sqrtNbPoints = std::size_t(std::sqrt(initData.nbPoints));
        std::size_t nbPointsY = initData.nbPoints / sqrtNbPoints;
        std::size_t nbPointsX = sqrtNbPoints + (initData.nbPoints % sqrtNbPoints > 0 ? 1 : 0);

        x_local = (rank - 1) * (nbPointsX / nbpCalc);
        nbPoints_local = nbPointsX * nbPointsY;
    }

    if (initData.modeGeneration == 0) // Génération sur toute la grille
    {

        fullConfig.cloud = Geometry::generatePointsIn(initData.nbPoints, nbPoints_local, {fullConfig.cartesianGrid.getLeftBottomVertex(), fullConfig.cartesianGrid.getRightTopVertex()}, x_local);
    }
    else
    {
        fullConfig.cloud = Geometry::generatePointsIn(initData.nbPoints, nbPoints_local, {point{initData.xl, initData.yb}, point{initData.xr, initData.yt}}, x_local);
    }

    Simulation::Vortices vortices(initData.nbVortices, {fullConfig.cartesianGrid.getLeftBottomVertex(),
                                                        fullConfig.cartesianGrid.getRightTopVertex()});

    fullConfig.vortices = vortices;

    for (std::size_t iVortex = 0; iVortex < initData.nbVortices; ++iVortex)
    {
        fullConfig.vortices.setVortex(iVortex, point{initData.x[iVortex], initData.y[iVortex]}, initData.force[iVortex]);
    }
}

int main(int nargs, char *argv[])
{
    /* ------------- Initialisation de MPI ----------------- */
    int rank, nbp;
    MPI_Comm globCom;

    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &globCom);
    MPI_Comm_size(globCom, &nbp);
    MPI_Comm_rank(globCom, &rank);

    /* --------------- Définition des Datatypes MPI ------------------ */
    MPI_Datatype geomVect_DT = create_geometry_vector_datatype<double>();   // Geometry::Vector<double>
    MPI_Datatype cloudPoint_DT = create_point_datatype<double>(MPI_DOUBLE); // Geometry::Point<double>

    // Datatype pour l'envoie des commandes de la simulation, du processus graphique aux autres processus
    MPI_Datatype simulationCommands_DT = create_simulationCommands_datatype();

    // Datatype pour l'envoie des données nécessaires à l'initialisation des classes
    MPI_Datatype initializationData_DT = create_initializationData_datatype();

    /* ------------- Variables nécessaires à la simulation ---------------- */
    // Variable contenant l'état complet de la simulation
    FullConfig fullConfig;

    // Données nécessaires à l'initialisation des classes
    InitializationData initData;

    // Variables pour l'affichage
    std::size_t resx = 800, resy = 600;

    /* ---------------------------------- Récupération des données de la simulation et affichage des instructions --------------------------------- */

    // Processus graphique
    if (rank == 0)
    {
        if (nargs > 3)
        {
            resx = std::stoull(argv[2]);
            resy = std::stoull(argv[3]);
        }

        std::cout << "######## Vortex simulator ########" << std::endl
                  << std::endl;
        std::cout << "Press P for play animation " << std::endl;
        std::cout << "Press S to stop animation" << std::endl;
        std::cout << "Press right cursor to advance step by step in time" << std::endl;
        std::cout << "Press down cursor to halve the time step" << std::endl;
        std::cout << "Press up cursor to double the time step" << std::endl;
    }
    else if (rank == 1) // Processus de calcul qui récupère les données nécessaires dans le fichier
    {
        char const *filename;
        if (nargs == 1)
        {
            std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;

            initData.dataLoaded = false;
        }
        else
        {
            filename = argv[1];
            std::ifstream fich(filename);
            initData = getConfig(fich);
            fich.close();

            initData.dataLoaded = true;
        }
    }

    /* --------------------------------------- Envoie des données de simulation à tous les processus ---------------------------------- */

    // Envoie des données nécessaires à l'initialisation des classes
    MPI_Bcast(&initData, 1, initializationData_DT, 1, globCom);

    // Arrêt si les données n'ont pas été chargées
    if (!initData.dataLoaded)
    {
        std::cout << "Data not properly loaded" << std::endl;

        if (initData.x != NULL)
            delete[] initData.x;
        if (initData.y != NULL)
            delete[] initData.y;
        if (initData.force != NULL)
            delete[] initData.force;

        StopMPI(&simulationCommands_DT, &initializationData_DT, &cloudPoint_DT, &geomVect_DT);

        return EXIT_FAILURE;
    }

    // Envoie des données des vortex aux processus
    if (rank != 1)
    {
        initData.x = new double[initData.nbVortices];
        initData.y = new double[initData.nbVortices];
        initData.force = new double[initData.nbVortices];
    }

    MPI_Bcast(initData.x, initData.nbVortices, MPI_DOUBLE, 1, globCom);
    MPI_Bcast(initData.y, initData.nbVortices, MPI_DOUBLE, 1, globCom);
    MPI_Bcast(initData.force, initData.nbVortices, MPI_DOUBLE, 1, globCom);

    // Création et initialisation des classes
    readConfig(initData, rank, nbp - 1, fullConfig);
    fullConfig.isMobile = initData.isMobile;

    /* --------------------------------------------------- Calcul et affichage de la simulation ---------------------------------------------------- */
    SimulationCommands simCommand;

    simCommand.animate = false;
    simCommand.dt = 0.1;
    simCommand.advance = false;
    simCommand.endSimulation = false;

    // On récupère la taille des données à transmettre
    int cloudPoint_DT_size, geomVect_DT_size;
    MPI_Type_size(cloudPoint_DT, &cloudPoint_DT_size);
    MPI_Type_size(geomVect_DT, &geomVect_DT_size);

    int grid_size_as_double = fullConfig.cartesianGrid.get_container_size() * geomVect_DT_size / sizeof(double);
    int cloud_size_as_double = fullConfig.cloud.numberOfPoints() * cloudPoint_DT_size / sizeof(double);
    int vortices_size = fullConfig.vortices.get_container_size();

    // On crée le buffer d'envoie/de réception des données
    int allBufferSize = grid_size_as_double + fullConfig.vortices.get_container_size() + cloud_size_as_double;

    double *gigaBuffer = NULL;
    if (fullConfig.isMobile)
        gigaBuffer = new double[allBufferSize];

    MPI_Request req[3];
    int nbReq = 0; // Nbr de requetes en cours

    // Processus graphique
    if (rank == 0)
    {
        std::size_t sqrtNbPoints = std::size_t(std::sqrt(initData.nbPoints));
        std::size_t nbPointsY = initData.nbPoints / sqrtNbPoints;
        std::size_t nbPointsX = sqrtNbPoints + (initData.nbPoints % sqrtNbPoints > 0 ? 1 : 0);
        int nbPoints = nbPointsX * nbPointsY;

        int *sizes = new int[nbp];
        sizes[0] = 0;
        for (int i = 1; i < nbp; i++)
            sizes[i] = nbPoints * cloudPoint_DT_size / sizeof(double);

        int *displ = new int[nbp];
        displ[0] = 0;
        for (int i = 1; i < nbp; i++)
        {
            displ[i] = displ[i - 1] + sizes[i - 1];
            std::cout << sizes[i] << " : " << cloud_size_as_double << std::endl;
        }

        // Reception de la grille initale
        MPI_Recv(fullConfig.cartesianGrid.data(), fullConfig.cartesianGrid.get_container_size(), geomVect_DT, 1, 101, globCom, MPI_STATUS_IGNORE);
        // MPI_Gatherv(NULL, 0, geomVect_DT, fullConfig.cartesianGrid.data(), sizes, displ, geomVect_DT, 0, globCom);

        Graphisme::Screen myScreen({resx, resy}, {fullConfig.cartesianGrid.getLeftBottomVertex(), fullConfig.cartesianGrid.getRightTopVertex()});

        while (myScreen.isOpen())
        {
            auto start = std::chrono::system_clock::now();
            simCommand.advance = false;

            // On inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
            sf::Event event;
            while (myScreen.pollEvent(event))
            {
                // évènement "fermeture demandée" : on ferme la fenêtre
                if (event.type == sf::Event::Closed)
                {
                    myScreen.close();

                    simCommand.endSimulation = true;
                }
                else if (event.type == sf::Event::Resized)
                {
                    // on met à jour la vue, avec la nouvelle taille de la fenêtre
                    myScreen.resize(event);
                }

                // Gestion des évènements clavier
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
                    simCommand.animate = true;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                    simCommand.animate = false;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
                    simCommand.dt *= 2;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
                    simCommand.dt /= 2;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
                    simCommand.advance = true;
            }
            std::cout << "before bcast" << std::endl;
            // Broadcast/Envoie de advance, animate et dt (s'ils ont changé) au(x) processus de calcul (ie envoie d'un SimulationCommand)
            MPI_Bcast(&simCommand, 1, simulationCommands_DT, 0, globCom);
            std::cout << "aftezr bcast" << std::endl;

            if (!simCommand.endSimulation)
            {
                // Récupération des informations de simulation: vortices, grid et cloud si isMobile = true, seulement cloud sinon
                if (fullConfig.isMobile)
                {
                    // Réception des données (c'est la partie lente qui limite les FPS)
                    std::cout << "before bcast" << std::endl;
                    MPI_Ibcast(gigaBuffer, vortices_size, MPI_DOUBLE, 1, globCom, &req[0]);
                    std::cout << "before recv" << std::endl;
                    MPI_Irecv(&gigaBuffer[vortices_size], grid_size_as_double, MPI_DOUBLE, 1, 5, globCom, &req[1]);
                    std::cout << "before gather" << std::endl;

                    MPI_Igatherv(NULL, 0, MPI_DOUBLE, &gigaBuffer[vortices_size + grid_size_as_double], sizes, displ, MPI_DOUBLE, 0, globCom, &req[2]);

                    nbReq = 3;

                    std::cout << "after gather" << std::endl;

                    MPI_Waitall(nbReq, req, MPI_STATUSES_IGNORE);

                    std::cout << "after waitall graphique" << std::endl;

                    // Copie depuis le buffer dans la structure
                    std::copy(gigaBuffer, gigaBuffer + vortices_size, fullConfig.vortices.data());
                    std::copy(&gigaBuffer[vortices_size], &gigaBuffer[vortices_size] + grid_size_as_double, fullConfig.cartesianGrid.data());
                    std::copy(&gigaBuffer[vortices_size + grid_size_as_double], &gigaBuffer[vortices_size + grid_size_as_double] + cloud_size_as_double, fullConfig.cloud.data());
                }
                else
                {
                    std::cout << "before gathervast" << std::endl;
                    MPI_Igatherv(NULL, 0, MPI_DOUBLE, fullConfig.cloud.data(), sizes, displ, MPI_DOUBLE, 0, globCom, &req[0]);
                    std::cout << "after gathervast" << std::endl;

                    nbReq = 1;

                    MPI_Waitall(nbReq, &req[0], MPI_STATUSES_IGNORE);
                }

                // Mise à jour de la fenêtre
                myScreen.clear(sf::Color::Black);
                std::string strDt = std::string("Time step : ") + std::to_string(simCommand.dt);
                myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second - 96)});
                myScreen.displayVelocityField(fullConfig.cartesianGrid, fullConfig.vortices);
                myScreen.displayParticles(fullConfig.cartesianGrid, fullConfig.vortices, fullConfig.cloud);
                auto end = std::chrono::system_clock::now();
                std::chrono::duration<double> diff = end - start;
                std::string str_fps = std::string("FPS : ") + std::to_string(1. / diff.count());
                myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second - 96)});
                myScreen.display();
            }
        }

        if (sizes != NULL)
            delete[] sizes;
        if (displ != NULL)
            delete[] displ;
    }
    // Processus de calcul
    else
    {
        // Initialisation de la grille
        fullConfig.cartesianGrid.updateVelocityField(fullConfig.vortices);

        // Envoi asynchrone de la grille au processus graphique (Buffer inutile ici car on ne modifie pas cette donnée avant le prochain MPI_Wait)
        if (rank == 1)
        {
            MPI_Isend(fullConfig.cartesianGrid.data(), fullConfig.cartesianGrid.get_container_size(), geomVect_DT, 0, 101, globCom, &req[0]);
            nbReq = 1;
        }
        // MPI_Gatherv(fullConfig.cartesianGrid.data(), fullConfig.cartesianGrid.get_container_size(), geomVect_DT, NULL, NULL, NULL, geomVect_DT, 0, globCom);

        // On crée le buffer permettant un envoie asynchrone des données
        double *cloud_buffer = NULL;
        if (!fullConfig.isMobile)
            cloud_buffer = new double[cloud_size_as_double];

        // Tant que la simulation est en cours
        while (!simCommand.endSimulation)
        {
            std::cout << "before iggggggcast" << std::endl;
            // Récupération des commandes du processus graphique
            MPI_Bcast(&simCommand, 1, simulationCommands_DT, 0, globCom);
            std::cout << "afyer ibcadq<dfqzrfst" << std::endl;

            if (!simCommand.endSimulation)
            {
                std::cout << "before wait all" << std::endl;
                // On s'assure que les données précédentes ont bien été envoyées et réceptionnées
                MPI_Waitall(nbReq, req, MPI_STATUSES_IGNORE);
                std::cout << "after wait all" << std::endl;

                // Envoie asynchrone des données de la simulation au processus graphique: vortices, grid et cloud si isMobile, cloud sinon
                // On utilise des buffer pour s'assurer de ne pas modifier les données pendant l'envoi
                if (fullConfig.isMobile)
                {
                    // Copie dans le buffer
                    std::copy(fullConfig.vortices.data(), fullConfig.vortices.data() + vortices_size, gigaBuffer);
                    std::copy(fullConfig.cartesianGrid.data(), fullConfig.cartesianGrid.data() + grid_size_as_double, &gigaBuffer[vortices_size]);
                    std::copy(fullConfig.cloud.data(), fullConfig.cloud.data() + cloud_size_as_double, &gigaBuffer[vortices_size + grid_size_as_double]);

                    std::cout << "before ibcast" << std::endl;

                    // Envoie asynchrone
                    MPI_Ibcast(gigaBuffer, vortices_size, MPI_DOUBLE, 1, globCom, &req[0]);

                    std::cout << "after ibcast" << std::endl;

                    if (rank == 1)
                        MPI_Isend(&gigaBuffer[vortices_size], grid_size_as_double, MPI_DOUBLE, 0, 5, globCom, &req[2]);

                    MPI_Igatherv(&gigaBuffer[vortices_size + grid_size_as_double], cloud_size_as_double, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, globCom, &req[1]);

                    nbReq = rank == 1 ? 3 : 2;
                }
                else
                {
                    // Copie dans le buffer
                    std::copy(fullConfig.cloud.data(), fullConfig.cloud.data() + cloud_size_as_double, cloud_buffer);

                    // Envoie asynchrone
                    MPI_Igatherv(cloud_buffer, cloud_size_as_double, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, globCom, &req[0]);

                    nbReq = 1;
                }

                // Mesure de la vitesse de calcul
                auto start = std::chrono::system_clock::now();

                // Simulation
                if (simCommand.animate | simCommand.advance)
                {
                    if (fullConfig.isMobile)
                    {
                        fullConfig.cloud = Numeric::solve_RK4_movable_vortices(simCommand.dt, fullConfig.cartesianGrid, fullConfig.vortices, fullConfig.cloud, rank);
                    }
                    else
                    {
                        fullConfig.cloud = Numeric::solve_RK4_fixed_vortices(simCommand.dt, fullConfig.cartesianGrid, fullConfig.cloud);
                    }
                }

                auto end = std::chrono::system_clock::now();

                std::chrono::duration<double> diff = end - start;
                std::string str_fps = std::string("Calculations per second  : ") + std::to_string(1. / diff.count());
                std::cout << str_fps << "\n";
            }
        }

        // Libération de la mémoire
        if (cloud_buffer != NULL)
            delete[] cloud_buffer;
    }

    // Libération de la mémoire et fin
    if (gigaBuffer != NULL)
        delete[] gigaBuffer;

    if (initData.x != NULL)
        delete[] initData.x;
    if (initData.y != NULL)
        delete[] initData.y;
    if (initData.force != NULL)
        delete[] initData.force;

    StopMPI(&simulationCommands_DT, &initializationData_DT, &cloudPoint_DT, &geomVect_DT);

    return EXIT_SUCCESS;
}