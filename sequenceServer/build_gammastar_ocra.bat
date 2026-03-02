docker build --tag gs-ocra -f gammastar_ocra/docker/Dockerfile .

CD gammastar_recon
CALL build_gs-recon.bat
pause