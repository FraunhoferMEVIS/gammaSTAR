<div align="center">
  <img src="img/gammaSTAR.png" alt="gammaSTAR" width="300"/>
  <img src="img/Resonint.png" alt="Resonint" width="150"/>
  </div>

# Release of gammaSTAR ilumr
This set of tools enables the local installation of a docker container that allows to use the ilumr MR tabletop system from Resonint and an automatic image reconstruction based on gammaSTAR raw representations.

## License
This software is distributed under the GNU Affero General Public License v3 (AGPL v3).  

## Prerequisites
1. Have [Docker Desktop](https://docs.docker.com/desktop/) or similar Docker variants installed. 
2. Follow the instructions of the sequence server installation
3. Configure the ethernet port:
   - IP Address: `192.168.137.1`
   - Subnet Mask: `255.255.255.0`
   - Gateway: `192.168.137.4`
   - Preferred DNS: `4.4.4.4`
4. Run the "Find Frequency" and "Autoshim" programs of the ilumr jupyter environment

## Installation of ilumr system code on the ilumr tabletop MRI system
Copy this directory onto ilumr system via jupyterlab interface
ssh into ilumr by doing this: 
	open Xcode or another terminal
	run: ssh xilinx@192.168.137.2
	password: xilinx

and navigate to this directory, then run these commands:

	sudo copy docker-compose.yml /etc/matipo/docker-compose.yml
	docker compose -f /etc/matipo/docker-compose.yml pull
	sudo systemctl restart matipo

## How to use after installation
ssh into ilumr and run:

	docker compose -f /etc/matipo/docker-compose.yml exec lua bash

This will open a new shell in the ilumr-lua container.
Navigate to the directory with the `main.lua` file by running these commands:
	cd ..
	cd home/ilumr_system_code/Sequence_processor

And execute the `main.lua` file with:
	luajit main.lua

The output should look like this:

	Mon Jan  5 13:33:00 2026 DEBUG DMA_SIZE: 0x8000000
	Mon Jan  5 13:33:00 2026 DEBUG DMA_DATA_OFFSET: 0x1000000
	Mon Jan  5 13:33:00 2026 DEBUG DMA_DATA_SIZE: 0x7000000
	Mon Jan  5 13:33:00 2026 INFO Connected exec_req to tcp://driver:5005
	Mon Jan  5 13:33:00 2026 INFO Connected exec_push to tcp://driver:5006
	Mon Jan  5 13:33:00 2026 INFO Connected exec_sub to tcp://driver:5007
	Mon Jan  5 13:33:00 2026 INFO TCP server started on 0.0.0.0:8765


It may also be useful to monitor the status of the driver container using this command:

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