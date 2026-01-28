docker build --tag gstar_ocra -f gammastar_ocra/docker/Dockerfile .

CD gammastar_recon
CALL build_gammaSTAR_recon.bat
pause