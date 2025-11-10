REM
FOR /F "tokens=1-4 delims=/ " %%a IN ("%date%") DO (
	SET BUILD_DATE=%%a-%%b-%%c
)
FOR /F "tokens=1-2 delims=: " %%a IN ("%time%") DO (
	SET BUILD_TIME=%%a-%%b
)
SET BUILD_DATETIME=%BUILD_DATE%_%BUILD_TIME%
CD ..
docker build . -t registry.fme.lan/gammastar/test_client_gstar_recon_release --build-arg buildDateTime=%BUILD_DATETIME% -f tests/Dockerfile