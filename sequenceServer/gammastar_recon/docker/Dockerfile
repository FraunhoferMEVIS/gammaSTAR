ARG base_image_cpu_runtime=ubuntu:22.04

### Runtime stage for cpu applications ####
FROM $base_image_cpu_runtime AS gstar_recon

RUN apt-get dist-upgrade && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests --yes pip ffmpeg libsm6 libxext6

COPY docker/main_gstar_recon.py /opt/code/main_gstar_recon.py
COPY third_party_licenses/ /opt/code/third_party_licenses

COPY mrpy_tools/reconstruction/mrpy_recon_tools /opt/code/mrpy_recon_tools
RUN pip install /opt/code/mrpy_recon_tools

COPY modules /opt/code/modules
RUN pip install /opt/code/modules

WORKDIR /opt/code
CMD ["python3", "-u", "main_gstar_recon.py"]
EXPOSE 9002