<div align="center">
  <img src="img/gammaSTAR.png" alt="gammaSTAR" width="300"/>
  <img src="img/Resonint.png" alt="Resonint" width="150"/>
  </div>

# Release of gammaSTAR ilumr
This set of tools enables the local installation of a docker container that allows to use the ilumr MR tabletop system from Resonint and an automatic image reconstruction based on gammaSTAR raw representations.

## License
This software is distributed under the GNU Affero General Public License v3 (AGPL v3).  

## Prerequisites
1. Install [Docker Desktop](https://docs.docker.com/desktop/) or similar Docker variant 
2. Follow the instructions of the gammaSTAR sequence server installation on your PC
3. Connect directly from your PC to ilumr using an ethernet cable
4. Connect your PC to a Wifi network with internet access
5. Enable Internet Connection Sharing from the Wifi adapter to the Ethernet adapter, see https://superuser.com/a/1899168
6. With ilumr powered on, access the jupyterlab interface at 192.168.137.2 in your browser.
7. Insert the shim sample and run the "Find Frequency" and "Autoshim" programs of the ilumr jupyter environment

## Installation of gammaSTAR ilumr driver on the ilumr tabletop MRI system
Open a terminal in the JupyterLab interface (ctrl+shift+L -> Terminal) and clone the gammastar repository to the `/home` directory:
	
	cd /home
	git clone https://github.com/FraunhoferMEVIS/gammaSTAR.git

Then run the install script:

	cd gammaSTAR/sequenceServer/gammastar_ilumr/ilumr_system_code/
	bash install.sh

When prompted, type the system's ssh password and press enter (default: xilinx).

This will take some time to download the docker images.

## How to use after installation
The container will be run automatically, and the gammaSTAR ilumr server should be able to connect and run sequences. If there are issues the logs may be viewed by logging into the host OS with ssh using a jupyterlab terminal (default password: xilinx):

	ssh xilinx@host.docker.internal

Then the logs can be viewed with:

	docker compose -f /etc/matipo/docker-compose.yml logs -f gammastar

The output should look like this:

	Mon Jan  5 13:33:00 2026 DEBUG DMA_SIZE: 0x8000000
	Mon Jan  5 13:33:00 2026 DEBUG DMA_DATA_OFFSET: 0x1000000
	Mon Jan  5 13:33:00 2026 DEBUG DMA_DATA_SIZE: 0x7000000
	Mon Jan  5 13:33:00 2026 INFO Connected exec_req to tcp://driver:5005
	Mon Jan  5 13:33:00 2026 INFO Connected exec_push to tcp://driver:5006
	Mon Jan  5 13:33:00 2026 INFO Connected exec_sub to tcp://driver:5007
	Mon Jan  5 13:33:00 2026 INFO TCP server started on 0.0.0.0:8765

It may also be useful to monitor the status of the ilumr driver container using this command, to confirm pulse sequences are being executed:

	docker compose -f /etc/matipo/docker-compose.yml logs driver -f

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