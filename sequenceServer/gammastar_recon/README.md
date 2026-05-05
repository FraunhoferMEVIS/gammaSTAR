<div align="center">
  <img src="img/logo.png" alt="logo" width="300"/>
</div>

# gammaSTAR Reconstructions v1.1.1 Sim. Only Version
A fully functional reconstruction server which is compatible with the publicly available gammaSTAR frontend. Simulation of 2D/3D Cartesian/non-Cartesian sequences is supported. Examplary demonstration of usage in combination with tabletop systems, MR simulators and full body systems is given in
> Huber J, Hussain S, Kuhlen V, Neisser A, Schenk L, Konstandin S, Günther M, Klimenko M, Hoinkiss D. Introducing gammaSTAR Reconstructions: Demonstration of Vendor Neutral MR Data Acquisition and Reconstruction on Tabletop Systems, MR Simulators and 3T Systems. Proceedings of the 41st Annual Meeting of the ESMRMB.

## License
This software is distributed under the GNU Affero General Public License v3 (AGPL v3).  
A commercial license is available for organizations or individuals requiring alternative licensing terms. For commercial licensing inquiries, please contact the [project maintainers](#contributors). A commercial license can include additional features developed by Fraunhofer MEVIS such as
- Full access to the Fraunhofer MEVIS reconstruction library, including solutions for parallel imaging, partial Fourier etc.
- Prospective and retrospective motion correction solutions
- Automatic quality control and analysis of acquired data
- Patient adaptive solutions for improved image quality
- Certified solutions (IEC 62304) which are compatible to the Open Recon interface by Siemens Healthineers
- Integration of your own reconstruction or processing algorithms into clinical workflows

## Prerequisites
Have [Docker Desktop](https://docs.docker.com/desktop/) installed. 

## Building gammaSTAR Reconstructions
The gammastar reconstruction docker image can be built using the following command from the gammastar_recon directory
```bash
docker build . --target=gs-recon -t gs-recon -f docker/Dockerfile
```
Alternatively, double-click the `build_gs-recon.bat` file. <br>

## Running gammaSTAR Reconstructions
The reconstruction server needs to run alongside the sequence server in order to simulate and reconstruct sequences from the gammaSTAR frontend. Therefore, navigate to the the "sequenceServer" folder and from a terminal run
```bash
docker compose -f compose_server.yaml up
```

## Unit Tests
The software comes with unit tests which validate the correct functionality of the underlying mrpy_recon_tools units. For execution you need to perform the following commands from the command line interface
```bash
pip install coverage
pip install mrpy_tools/reconstruction/mrpy_recon_tools
coverage run -m --source=. unittest discover -s mrpy_tools/reconstruction/mrpy_recon_tools/tests/unit_tests
coverage report
```
The code coverage can be found [`here`](mrpy_tools/reconstruction/mrpy_recon_tools/tests/coverage.txt).

## Contributors
This project is maintained by [Jörn Huber](https://www.mevis.fraunhofer.de/en/employees/joern-huber.html)

Thanks to the following contributors for their valuable input:

- Tom Lütjen
- [Daniel Hoinkiss](https://www.mevis.fraunhofer.de/en/employees/daniel-hoinkiss.html) 
- Vincent Kuhlen
- Arne Neisser