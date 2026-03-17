#include <iostream>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>

#include "dustdynamics.h"
#include "sources.h"
#include "cuda_array.h"
#include "grid.h"
#include "field.h"
#include "constants.h"
#include "gas1d.h"
#include "hydrostatic.h"
#include "file_io.h"
#include "errorfuncs.h"

#include "coagulation/coagulation.h"
#include "coagulation/integration.h"
#include "coagulation/fragments.h"

void compute_cs2(const Grid &g, Field<double> &T, Field<double> &cs2, double mu) {

    // Calculates square of the sound speed

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            cs2(i,j) = R_gas * T(i,j) / mu;
        }
    }
}

void compute_nu(const Grid &g, CudaArray<double> &nu, Field<double> &cs2, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        nu[i] = alpha * cs2(i,2) / Om;
    }
}

void compute_D(const Grid &g, Field3D<double> &D, Field<Prims> &wg, Field<double> &cs2, double Mstar, double delta, double Sc) {

    // Calculates the dust diffusion constant

    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        double Om = std::sqrt(GMsun * Mstar / (g.Rc(i)*g.Rc(i)*g.Rc(i)));
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            for (int k=0; k<D.Nd; k++) {
                D(i,j,k) = wg(i,j).rho * delta * cs2(i,j) / (Sc*Om) ;
            }
        }
    }
}

void compute_nu(const Grid &g, CudaArray<double> &nu, double nu0, double Mstar, double alpha) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        nu[i] = alpha * nu0 * (g.Rc(i)/au) / std::sqrt(Mstar);
    }
}

void cs2_to_cs(Grid& g, Field<double> &cs, Field<double> &cs2) {
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            cs(i,j) = std::sqrt(cs2(i,j));
        }
    }
}

int main() {

    std::filesystem::path dir = std::string("./codes/outputs/PowerLawDisk_alpha1e-4_DPdv");
    std::filesystem::create_directories(dir);

    // Set up spatial grid 

    Grid::params p;
    p.NR = 200;
    p.Nphi = 100;
    p.Nghost = 2;

    p.Rmin = 5.*au;
    p.Rmax = 50.*au;
    p.theta_min = 0. ;
    p.theta_power = 0.75;
    p.theta_max = 0.2;

    p.R_spacing = RadialSpacing::log ;
    p.theta_spacing = ThetaSpacing::power;

    bool eps_profile = false;

    double  mu = 2.34, 
    M_star = 1., 
    alpha = 1e-4, 
    delta = 1e-4, 
    T_star=4397., 
    R_star = 3.096*Rsun,
    r_c = 50.*au,
    Mdisc = 0.0508772*Msun,
    pSig = -1.,
    d_to_g = 0.01,
    Sc = 1.,
    rho_p = 1.0,
    a0 = 5e-5,
    a1 = 50.,
    aini = 1e-4,
    v_frag = 100.;
    double L_star = 4.*M_PI*sigma_SB*std::pow(T_star, 4.)*std::pow(R_star, 2.);
    int n_spec = 7.*3.*std::log10(a1/a0) + 1;
    Grid g(p);

    // Setup a size distribution

    std::cout << "Number of dust species: "<< n_spec << "\n";
    SizeGrid sizes(a0, a1, n_spec, rho_p) ;

    write_grids(dir, &g, &sizes); // Write grids to file

    // Create star

    Star star(GMsun*M_star, L_star, T_star);

    // Create gas and dust fields

    Field3D<Prims> Ws_d = create_field3D<Prims>(g, n_spec); // Dust quantities 
    Field<Prims> Ws_g = create_field<Prims>(g); // Gas primitives
    CudaArray<double> Sig_g = make_CudaArray<double>(g.NR+2*g.Nghost); // Gas surface density
    CudaArray<double> nu = make_CudaArray<double>(g.NR+2*g.Nghost); // Kinematic viscosity
    Field<double> T = create_field<double>(g); // Temperature
    Field<double> J = create_field<double>(g); // Dummy J
    Field<double> cs = create_field<double>(g); // Sound speed
    Field<double> cs2 = create_field<double>(g); // Sound speed squared
    Field<double> alpha2D = create_field<double>(g); // alpha 2D
    Field<double> delta2D = create_field<double>(g); // delta 2D
    Field3D<double> D = create_field3D<double>(g, n_spec); // Dust diffusion constant 

    // ------------------------------------------------------------
    // ------------- Set up initial gas variables -----------------
    // ------------------------------------------------------------
    double Sigma0 = (2.+pSig)*Mdisc/(2.*M_PI*r_c*r_c) * std::pow(au/r_c, pSig) * std::exp(-std::pow(au/r_c, 2.+pSig));
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        // Set up gas
        Sig_g[i] =  Sigma0 * pow(g.Rc(i)/au, pSig);
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            T(i,j) = std::pow(0.5*0.05 * star.L / (4. * M_PI * g.Rc(i)*g.Rc(i) * sigma_SB), 0.25);
            cs(i,j) = std::sqrt(k_B*T(i,j) / (mu*m_H));
            cs2(i,j) = k_B*T(i,j) / (mu*m_H);
            nu[i] = alpha * cs(i,j) * cs(i,j) / std::sqrt(star.GM/std::pow(g.Rc(i), 3.));
        }
    }
    // ------------------------------------------------------------

    double M_gas=0, M_dust=0;
    for (int i=g.Nghost; i<g.NR+g.Nghost; i++ ) { M_gas += Sig_g[i]*2.*M_PI*g.Rc(i)*g.dRe(i);}
    std::cout << "Initial gas mass: " << M_gas/Msun << " M_sun\n";
        
    int gas_boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;
    double gas_floor = 1e-100;
    double floor = 1.e-10;

    compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, gas_floor);
    calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);   
    for (int i=0; i<g.NR + 2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi + 2*g.Nghost; j++) {
            alpha2D(i,j) = alpha;
            delta2D(i,j) = delta;
        }
    }

    // ------------------------------------------------------------
    // -------------- Set up initial dust disk --------------------
    // ------------------------------------------------------------
    for (int i=0; i<g.NR+2*g.Nghost; i++) {
        for (int j=0; j<g.Nphi+2*g.Nghost; j++) {
            double rho_tot = 0;
            for (int k=0; k<Ws_d.Nd; k++) {
                double St_mid = 0.5*M_PI*sizes.centre_size(k)*rho_p/Sig_g[i]; // midplane Stokes number
                double vk = std::sqrt(GMsun*M_star/g.Rc(i));
                double H  = cs(i,j)/vk * g.Rc(i);
                double Hd = H * std::sqrt(delta/(St_mid+delta));
                // Initialise dust with MRN profile and exponential cut off at 0.1 micron
                double a_i   = std::pow(3*sizes.edge_mass(k)/(4*M_PI*rho_p), 1/3.);
                double a_ip1 = std::pow(3*sizes.edge_mass(k+1)/(4*M_PI*rho_p), 1/3.);
                double qp4 = -3.5 + 4.;
                
                double eps;
                if(a_ip1<aini)
                    eps = d_to_g * (std::pow(a_ip1, qp4) - std::pow(a_i, qp4))/(std::pow(aini, qp4)-std::pow(a0, qp4));
                else if(a_ip1>aini && a_i<aini)
                    eps = d_to_g * (std::pow(aini, qp4)  - std::pow(a_i, qp4)) /(std::pow(aini, qp4)-std::pow(a0, qp4));
                else
                    eps = floor;

                double den;
                if(eps_profile){
                    double rhod_mid = eps*Sig_g[i] / (std::sqrt(2.*M_PI) * Hd);
                    den = rhod_mid * std::exp(-St_mid/delta*(std::exp(0.5*std::pow(g.Zc(i,j)/H,2)) - 1.) - 0.5*std::pow(g.Zc(i,j)/H,2));
                } else {
                    den = eps * Ws_g(i,j).rho;
                }
                
                Ws_d(i,j,k).rho = den;
                if (Ws_d(i,j,k).rho <= Ws_g(i,j).rho*floor) {
                    Ws_d(i,j,k).rho = Ws_g(i,j).rho*floor;
                }

                rho_tot += Ws_d(i,j,k).rho;
                D(i,j,k) = Ws_g(i,j).rho * (delta * cs(i,j) * cs(i,j) / std::sqrt(GMsun/std::pow(g.Rc(i), 3.))) / Sc ;

                Ws_d(i,j,k).v_R   = 0.;
                Ws_d(i,j,k).v_phi = vk;
                Ws_d(i,j,k).v_Z   = 0.;
            }
        }
    }
    // ------------------------------------------------------------

    for (int i=g.Nghost; i<g.NR + g.Nghost; i++) {
        for (int j=g.Nghost; j<g.Nphi + g.Nghost; j++) { 
            for (int k=0; k<Ws_d.Nd; k++) {
                M_dust += 4.*M_PI * Ws_d(i,j,k).rho * g.volume(i,j); // 4pi comes from 2pi in azimuth and 2 for symmetry about midplane
            }
        }
    }

    // Set up coagulation kernel, storing the fragmentation velocity

    BirnstielKernel kernel = BirnstielKernel(g, sizes, Ws_d, Ws_g, cs, delta2D, mu, M_star);
    kernel.set_fragmentation_threshold(v_frag);

    // Setup the integrator
    BS32Integration<CoagulationRate<decltype(kernel), SimpleErosion>>
        coagulation_integrate(
            create_coagulation_rate(
                sizes, 
                kernel, 
                SimpleErosion(1,11/6.,sizes.min_mass())), 
            1e-2, 1e-10
        ) ;

    std::cout << "Initial dust mass: " << M_dust/Msun << " M_sun\n";

    // Choose times to store data
    
    double t = 0, dt;
    const int ntimes = 200;  
    double ts[ntimes+1];
    double tstart = 100.*year;
    double tend   = 2e5*year;
    double log_dt = std::log(tend/tstart)/(ntimes);
    for(int i=0; i<ntimes+1; i++){
        ts[i] = tstart * std::exp(i*log_dt);
    }

    std::ofstream f_times((dir / "2Dtimes.txt"));
    f_times << 0. << "\n";
    for (int i=0; i<ntimes+1; i++) {
        f_times << ts[i] << "\n";
    } 
    f_times.close();

    // Initialise diffusion-advection solver

    Sources src(T, Ws_g, sizes, floor, M_star, mu);
    DustDynamics dyn(D, cs, src, 0.4, 0.2, floor, gas_floor);

    double dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);

    std::cout << dt_CFL << "\n";

    // Set up boundary conditions

    int boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer | BoundaryFlags::open_Z_outer;

    dyn.set_boundaries(boundary);

    std::chrono::_V2::system_clock::time_point start,stop;
    start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds duration;
    double yps;
    int count = 0;
    double t_coag = 0, dt_coag = 0, t_temp = 0, dt_1perc = year;

    dt_CFL = 1;

    int Nout = 1;

    double dummy = 0;

    std::ifstream f(dir / ("restart_params.dat"), std::ios::binary);
    double t_restart=0;

    if (f) {

        // This block is used for reading in restart configurations if running on a cluster that requires restarting the code
        
        read_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, dummy);

        std::cout << "Restart params: " << count << " " << t/year << " " << dt_CFL/year << "\n";

        read_restart_prims(dir, Ws_d, Ws_g, Sig_g);

        compute_cs2(g,T,cs2,mu);
        cs2_to_cs(g, cs, cs2);
        compute_D(g, D, Ws_g, cs2, M_star, delta, Sc);
        compute_nu(g, nu, cs2, M_star, alpha);
        t_restart = t;
    }
    else {
        compute_nu(g, nu, cs2, M_star, alpha);
        compute_D(g, D, Ws_g, cs2, M_star, delta, Sc);
        write_prims(dir, 0, g, Ws_d, Ws_g, Sig_g);
        write_temp(dir, 0, g, T) ; 
    }


    // Main timestep iteration

    for (double ti : ts) {

        if (t > ti) {
            Nout += 1;
            continue;
        }

        while (t < ti) {    

            if (!(count%1000)) {
                std::cout << "t = " << t/year << " years\n";
                std::cout << "dt = " <<dt_CFL/year << " years\n";
                stop = std::chrono::high_resolution_clock::now();
                yps = ((t-t_restart)/year) / std::chrono::duration_cast<std::chrono::seconds>(stop - start).count();
                std::cout << "Years per second: " << yps << "\n";
            }

            dt = std::min(dt_CFL, ti-t); // Set time-step according to CFL condition or proximity to selected time snapshots
            
            dyn(g, Ws_d, Ws_g, dt); // Diffusion-advection update

            // Gas updates
            // update_gas_sigma(g, Sig_g, dt, nu, gas_boundary, gas_floor);
            // compute_hydrostatic_equilibrium(star, g, Ws_g, cs2, Sig_g, Ws_d, gas_floor);
            // calc_gas_velocities(g, Sig_g, Ws_g, cs2, nu, alpha, star, gas_boundary, gas_floor);  
            compute_D(g, D, Ws_g, cs2, M_star, delta, Sc);

            // Coagulation update when 1 internal coagulation time-step has passed in the global simulation time
            if ((t+dt >= t_coag+dt_coag)|| (t+2*dt >= t_coag+dt_coag && dt < dt_coag) || dt == ti-t) {
                std::cout << "Coag step at count = " << count << "\n";
                // Run coagulation internal integration (routine calculates its own sub-steps to integrate over the timestep passed into it)
                coagulation_integrate.integrate(g, Ws_d, Ws_g, (t+dt)-t_coag, dt_coag, floor) ;
                t_coag = t+dt;
            } 

            count += 1;
            t += dt;

            if (count < 1000) {
                dt_CFL = std::min(dyn.get_CFL_limit(g, Ws_d, Ws_g), 1.025*dt); // Calculate new CFL condition time-step 
            }
            else {
                dt_CFL = dyn.get_CFL_limit(g, Ws_d, Ws_g);
            }
                
            // Uncomment this section for writing restart files for jobs on clusters that need to be re-batched after a certain amount of time; here a restart file is written after 20 hrs
            // if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count()/3600. > 20.) {
            //     std::cout << "Writing restart at t = " << t/year << " years.\n" ;
            //     write_restart_file(dir / ("restart_params.dat"), count, t, dt_CFL, t_coag, t_temp, dt_coag, dt_1perc, dummy);
            //     write_restart_prims(dir, g, Ws_d, Ws_g, Sig_g);  
            //     return 0;
            // } 

        }

        // Record densities to file at time snapshots

        write_prims(dir, Nout, g, Ws_d, Ws_g, Sig_g);  
        //write_temp(dir, Nout, g, T) ; Skip because constant
        Nout+=1;
    }
    
    // This is used for telling your job submission script that the final snapshot has been reached, meaning no more restarts are necessary
    std::ofstream fin(dir / ("finished"));
    fin.close();

    stop = std::chrono::high_resolution_clock::now();
    std::cout << count << " timesteps\n" ;
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count()/(1.e6*60.) << " mins" << std::endl;  
    return 0;
} 
