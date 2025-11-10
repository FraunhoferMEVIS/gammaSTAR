ARG base_image=ubuntu:22.04

### Runtime stage for cpu applications ####
FROM $base_image AS test_client_gstar_recon

# Install dependencies
RUN apt-get dist-upgrade && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests --yes pip ffmpeg libsm6 libxext6

# Recon Tools Submodule
COPY mrpy_tools/reconstruction/mrpy_recon_tools /opt/code/mrpy_recon_tools
RUN pip install -e /opt/code/mrpy_recon_tools

# Copy Test Data into container
ARG buildDateTime="N/A"
RUN echo $buildDateTime > /opt/code/build_date_time.txt
COPY tests/testdata_release /opt/code/testdata
COPY tests/run_integration_tests.py /opt/code/run_integration_tests.py
COPY clients/mrd_to_dicom_client.py /opt/code/clients/mrd_to_dicom_client.py

# Test Execution
WORKDIR /opt/code
CMD ["python3", "-u", "run_integration_tests.py", "-f", "/opt/code/testdata"]
EXPOSE 9002