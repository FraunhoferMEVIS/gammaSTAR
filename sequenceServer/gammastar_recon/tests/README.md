# gammaSTAR Reconstructions Test Suite
To validate the reconstruction server with a library of available sequences, we provide a test client as a docker image which can be used for that. In order to build the client, you need to setup a folder which is filled with your available testdata in .h5 format. 

## Building the Test Suite
Double click the "build_test_container.bat" file. Alternatively pull the latest container with testdata from the registry using
```bash
docker pull registry.fme.lan/gammastar/test_client_gstar_recon
```

## Running the Test Suite with a gammaSTAR Reconstructions Image
In order to allow communication between the client and the server, you need to create a network first:
```bash
docker network create gammastar-net
```
Afterwards, open a terminal window and run the built gammaSTAR Reconstructions server:
```bash
docker run -p9002:9002 --network gammastar-net --name gstar_server -e FEEDBACK_HOST=host.docker.internal -e FEEDBACK_PORT=9003 --rm -it gstar_recon
```
Open another terminal (important: run it from the tests subdirectory or cd to that directory) and run the test suite image
```bash
docker run --network gammastar-net --name gstar_client -v ${pwd}\test_results:/opt/code/test_results --rm -it registry.fme.lan/gammastar/test_client_gstar_recon_release
```
