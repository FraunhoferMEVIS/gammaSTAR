docker build --tag gstar_ilumr -f gammastar_ilumr/docker/Dockerfile .

CD gammastar_recon
CALL build_gammaSTAR_recon.bat
pause