docker build --tag gs-server -f gammastar_server/docker/Dockerfile .

CD gammastar_recon
CALL build_gammaSTAR_recon.bat
pause