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

#define CONFIG_TYPE std::tuple<Simulation::Vortices, int, Numeric::CartesianGridOfSpeed, Geometry::CloudOfPoints>

struct SendConfig
{
    Simulation::Vortices vortices;
    int isMobile;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloud;
};

auto readConfigFile(std::ifstream &input)
{
    using point = Simulation::Vortices::point;

    int isMobile;
    std::size_t nbVortices;
    Numeric::CartesianGridOfSpeed cartesianGrid;
    Geometry::CloudOfPoints cloudOfPoints;
    constexpr std::size_t maxBuffer = 8192;
    char buffer[maxBuffer];
    std::string sbuffer;
    std::stringstream ibuffer;
    // Lit la première ligne de commentaire :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lecture de la grille cartésienne
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    double xleft, ybot, h;
    std::size_t nx, ny;
    ibuffer >> xleft >> ybot >> nx >> ny >> h;
    cartesianGrid = Numeric::CartesianGridOfSpeed({nx, ny}, point{xleft, ybot}, h);
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit mode de génération des particules
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    int modeGeneration;
    ibuffer >> modeGeneration;
    if (modeGeneration == 0) // Génération sur toute la grille
    {
        std::size_t nbPoints;
        ibuffer >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {cartesianGrid.getLeftBottomVertex(), cartesianGrid.getRightTopVertex()});
    }
    else
    {
        std::size_t nbPoints;
        double xl, xr, yb, yt;
        ibuffer >> xl >> yb >> xr >> yt >> nbPoints;
        cloudOfPoints = Geometry::generatePointsIn(nbPoints, {point{xl, yb}, point{xr, yt}});
    }
    // Lit le nombre de vortex :
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le nombre de vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    try
    {
        ibuffer >> nbVortices;
    }
    catch (std::ios_base::failure &err)
    {
        std::cout << "Error " << err.what() << " found" << std::endl;
        std::cout << "Read line : " << sbuffer << std::endl;
        throw err;
    }
    Simulation::Vortices vortices(nbVortices, {cartesianGrid.getLeftBottomVertex(),
                                               cartesianGrid.getRightTopVertex()});
    input.getline(buffer, maxBuffer); // Relit un commentaire
    for (std::size_t iVortex = 0; iVortex < nbVortices; ++iVortex)
    {
        input.getline(buffer, maxBuffer);
        double x, y, force;
        std::string sbuffer(buffer, maxBuffer);
        std::stringstream ibuffer(sbuffer);
        ibuffer >> x >> y >> force;
        vortices.setVortex(iVortex, point{x, y}, force);
    }
    input.getline(buffer, maxBuffer); // Relit un commentaire
    input.getline(buffer, maxBuffer); // Lit le mode de déplacement des vortex
    sbuffer = std::string(buffer, maxBuffer);
    ibuffer = std::stringstream(sbuffer);
    ibuffer >> isMobile;
    return std::make_tuple(vortices, isMobile, cartesianGrid, cloudOfPoints);
}

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

MPI_Datatype create_vortices_datatype(MPI_Datatype vorticesContainer_DT, MPI_Datatype geomVect_DT)
{
    int const count = 2;
    int blocklengths[count] = {1, 1};

    MPI_Datatype types[count] = {vorticesContainer_DT, geomVect_DT};
    MPI_Aint offsets[count] = {Simulation::Vortices::get_cotainer_offset(),
                               Simulation::Vortices::get_vector_offset()};

    MPI_Datatype vortices_DT;
    MPI_Type_create_struct(count, blocklengths, offsets, types, &vortices_DT);
    MPI_Type_create_resized(vortices_DT, 0, sizeof(Simulation::Vortices), &vortices_DT);
    MPI_Type_commit(&vortices_DT);

    return vortices_DT;
}

MPI_Datatype create_cartesian_grid_of_speed_datatype(MPI_Datatype cartesianContainer_DT)
{
    int count = 6;

    int block_lengths[count] = {1, 1, 1, 1, 1, 1};
    MPI_Aint displacements[count] = {*Numeric::CartesianGridOfSpeed::get_offsets()};
    MPI_Datatype types[count] = {MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, cartesianContainer_DT};

    MPI_Datatype cartesian_DT;
    MPI_Type_create_struct(count, block_lengths, displacements, types, &cartesian_DT);
    MPI_Type_create_resized(cartesian_DT, 0, sizeof(Numeric::CartesianGridOfSpeed), &cartesian_DT);
    MPI_Type_commit(&cartesian_DT);

    free(displacements);

    return cartesian_DT;
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

MPI_Datatype create_cloud_of_points_datatype(MPI_Datatype cloudContainer_DT)
{
    int count = 1;

    int block_lengths[count] = {1};
    MPI_Aint displacements[count] = {Geometry::CloudOfPoints::get_container_offset()};
    MPI_Datatype types[count] = {cloudContainer_DT};

    MPI_Datatype cloud_DT;
    MPI_Type_create_struct(count, block_lengths, displacements, types, &cloud_DT);
    MPI_Type_create_resized(cloud_DT, 0, sizeof(Geometry::CloudOfPoints), &cloud_DT);
    MPI_Type_commit(&cloud_DT);

    return cloud_DT;
}

MPI_Datatype create_config_datatype(MPI_Datatype vortices_DT, MPI_Datatype cartesianGrid_DT, MPI_Datatype cloud_DT)
{
    int count = 4;

    int block_lengths[count] = {1, 1, 1, 1};

    MPI_Aint displacements[count] = {
        offsetof(SendConfig, vortices),
        offsetof(SendConfig, isMobile),
        offsetof(SendConfig, cartesianGrid),
        offsetof(SendConfig, cloud)};

    MPI_Datatype types[count] = {vortices_DT, MPI_INT, cartesianGrid_DT, cloud_DT};

    MPI_Datatype config_DT;
    MPI_Type_create_struct(count, block_lengths, displacements, types, &config_DT);
    MPI_Type_create_resized(config_DT, 0, sizeof(SendConfig), &config_DT);
    MPI_Type_commit(&config_DT);

    return config_DT;
}

int main(int nargs, char *argv[])
{
    int rank, calcRank, nbp;
    MPI_Comm globCom, calcComm;

    MPI_Init(&nargs, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &globCom);
    MPI_Comm_size(globCom, &nbp);
    MPI_Comm_rank(globCom, &rank);

    // Initialisation du communicator de calcul
    if (rank != 0)
        MPI_Comm_split(globCom, 0, rank, &calcComm);

    MPI_Comm_rank(calcComm, &calcRank);

    // Processus graphique
    if (rank == 0)
    {
        std::size_t resx = 800, resy = 600;
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

        Graphisme::Screen myScreen({resx, resy}, {grid.getLeftBottomVertex(), grid.getRightTopVertex()});
        bool animate = false;
        double dt = 0.1;
    }
    else // Processus de calcul
    {
        // Vortices MPI datatype
        MPI_Datatype geomVect_DT = create_geometry_vector_datatype<double>();
        MPI_Datatype vorticesContainer_DT, vortices_DT;

        // Cartesian grid of speed MPI datatype
        MPI_Datatype cartesianContainer_DT, cartesianGrid_DT;

        // Cloud of points MPI datatype
        MPI_Datatype cloudPoint_DT = create_point_datatype<double>(MPI_DOUBLE);
        MPI_Datatype cloudContainer_DT, cloud_DT;

        // Final config datatype (tuple)
        MPI_Datatype config_DT;

        CONFIG_TYPE config;

        if (calcRank == 0)
        {
            char const *filename;
            if (nargs == 1)
            {
                std::cout << "Usage : vortexsimulator <nom fichier configuration>" << std::endl;
                return EXIT_FAILURE;
            }

            filename = argv[1];
            std::ifstream fich(filename);
            config = readConfigFile(fich);
            fich.close();

            auto vortices = std::get<0>(config);
            // auto isMobile = std::get<1>(config);
            auto grid = std::get<2>(config);
            auto cloud = std::get<3>(config);

            vorticesContainer_DT = create_vector_datatype<double>(MPI_DOUBLE, vortices.get_container_size());
            vortices_DT = create_vortices_datatype(vorticesContainer_DT, geomVect_DT);

            cartesianContainer_DT = create_vector_datatype<double>(MPI_DOUBLE, grid.get_container_size());
            cartesianGrid_DT = create_cartesian_grid_of_speed_datatype(cartesianContainer_DT);

            cloudContainer_DT = create_vector_datatype<Geometry::Point<double>>(cloudPoint_DT, cloud.get_container_size());
            cloud_DT = create_cloud_of_points_datatype(cloudContainer_DT);

            config_DT = create_config_datatype(vortices_DT, cartesianGrid_DT, cloud_DT);
        }

        MPI_Bcast(&config, 1, config_DT, 0, calcComm);

        auto vortices = std::get<0>(config);
        auto isMobile = std::get<1>(config);
        auto grid = std::get<2>(config);
        auto cloud = std::get<3>(config);

        grid.updateVelocityField(vortices);
    }

    while (myScreen.isOpen())
    {
        auto start = std::chrono::system_clock::now();
        bool advance = false;
        // on inspecte tous les évènements de la fenêtre qui ont été émis depuis la précédente itération
        sf::Event event;
        while (myScreen.pollEvent(event))
        {
            // évènement "fermeture demandée" : on ferme la fenêtre
            if (event.type == sf::Event::Closed)
                myScreen.close();
            if (event.type == sf::Event::Resized)
            {
                // on met à jour la vue, avec la nouvelle taille de la fenêtre
                myScreen.resize(event);
            }
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))
                animate = true;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                animate = false;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
                dt *= 2;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
                dt /= 2;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
                advance = true;
        }
        if (animate | advance)
        {
            if (isMobile)
            {
                cloud = Numeric::solve_RK4_movable_vortices(dt, grid, vortices, cloud);
            }
            else
            {
                cloud = Numeric::solve_RK4_fixed_vortices(dt, grid, cloud);
            }
        }
        myScreen.clear(sf::Color::Black);
        std::string strDt = std::string("Time step : ") + std::to_string(dt);
        myScreen.drawText(strDt, Geometry::Point<double>{50, double(myScreen.getGeometry().second - 96)});
        myScreen.displayVelocityField(grid, vortices);
        myScreen.displayParticles(grid, vortices, cloud);
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::string str_fps = std::string("FPS : ") + std::to_string(1. / diff.count());
        myScreen.drawText(str_fps, Geometry::Point<double>{300, double(myScreen.getGeometry().second - 96)});
        myScreen.display();
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}