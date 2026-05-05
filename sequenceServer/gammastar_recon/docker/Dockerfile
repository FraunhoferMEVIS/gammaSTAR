#Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
#AGPLv3-clause License
#
#The software is not qualified for use as a medical product or as part
#thereof. No bugs or restrictions are known.

ARG base_image_cpu_runtime=ubuntu:22.04

### Runtime stage for cpu applications ####
FROM $base_image_cpu_runtime AS gs-recon

RUN apt-get dist-upgrade && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests --yes pip ffmpeg libsm6 libxext6
RUN pip install --upgrade pip

COPY docker/main_gs-recon.py /opt/code/main_gs-recon.py
COPY third_party_licenses/ /opt/code/third_party_licenses
ARG buildDateTime="N/A"
RUN echo $buildDateTime > /opt/code/build_date_time.txt

COPY mrpy_tools/reconstruction/mrpy_recon_tools /opt/code/mrpy_recon_tools
RUN pip install /opt/code/mrpy_recon_tools

COPY modules /opt/code/modules
RUN pip install /opt/code/modules

WORKDIR /opt/code
CMD ["python3", "-u", "main_gs-recon.py"]
EXPOSE 9002