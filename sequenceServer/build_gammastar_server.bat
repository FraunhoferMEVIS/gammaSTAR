docker build --tag gstar_server -f gammastar_server/docker/Dockerfile .

CD gammastar_recon
CALL build_gammaSTAR_recon.bat
pause