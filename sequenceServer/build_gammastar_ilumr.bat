docker build --tag gs-ilumr -f gammastar_ilumr/docker/Dockerfile .

CD gammastar_recon
CALL build_gs-recon.bat
pause