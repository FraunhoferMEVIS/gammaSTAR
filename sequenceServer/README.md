<div align="center">
  <img src="img/gammaSTAR.png" alt="gammaSTAR" width="300"/>
</div>

# Combined Release of gammaSTAR Parser, gammaSTAR Server, gammaSTAR OCRA, gammaSTAR ilumr and gammaSTAR Reconstructions
This set of tools enables the local installation of a docker container that allows MR simulations using MR Zero (https://github.com/MRsources/MRzero-Core), controlling the OCRA tabletop MRI system (https://zeugmatographix.org/ocra/), controlling the Resonint ilumr tabletop MR systems (https://www.resonint.com/ilumr), and an automatic image reconstruction based on gammaSTAR raw representations.

## License
This software is distributed under the GNU Affero General Public License v3 (AGPL v3).  

## Prerequisites
1. Have [Docker Desktop](https://docs.docker.com/desktop/) or similar Docker variants installed. 

## How to Use
Use elevated command prompt
For running the simulation and reconstruction server:
1. Execute "build_gammastar_server.bat"
2  "docker compose up -f compose_server.yaml" inside this main directory

For running the OCRA and reconstruction server
1. Execute "build_gammastar_ocra.bat"
2  "docker compose up -f compose_ocra.yaml" inside this main directory

For running the ilumr and reconstruction server
1. Execute "build_gammastar_ilumr.bat"
2. "docker compose -f compose_ilumr.yaml up" inside this main directory
3. Install the ilumr_system_code on your tabletop system as described in ./gammastar_ilumr/ilumr_system_code/README.md

## Contributors
This project is maintained by [Daniel Christopher Hoinkiss](https://www.mevis.fraunhofer.de/en/employees/daniel-hoinkiss.html)

Thanks to the following contributors for their valuable input:

- Jörn Huber
- Simon Konstandin
- Vincent Kuhlen 
- Arne Neisser
- Tom Lütjen
- Snawar Hussain
- Lukas Schenk
- Juela Cufe