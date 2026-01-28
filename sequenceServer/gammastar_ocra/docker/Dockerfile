FROM python:3.12 AS build-base

# Install dependencies
RUN apt update

# Build wheel 
FROM build-base AS build

FROM python:3.12 AS server-base
RUN pip install fastapi[standard]

FROM server-base AS server
ADD gammastar_recon/mrpy_tools/reconstruction /src/reconstruction
RUN pip install -e /src/reconstruction/mrpy_recon_tools
ADD ./gammastar_recon /src/gammastar_recon

# Add sequenceServer
RUN pip install ismrmrd 
RUN pip install h5py
RUN pip install pydicom
COPY gammastar_ocra/sequenceServer.py /src/
RUN mkdir /data_export


# Add OCRA system code
RUN pip install msgpack
ADD gammastar_ocra/OCRA_system_code /src/OCRA_system_code
WORKDIR /src/OCRA_system_code
RUN git clone https://github.com/vnegnev/marcos_server.git
RUN git clone https://github.com/vnegnev/marcos_client.git
RUN mv local_config.py marcos_client/
RUN git clone https://github.com/vnegnev/marcos_extras.git
ENV PYTHONPATH="/src/OCRA_system_code/marcos_client"


# Expose the ports for both endpoints
EXPOSE 9000

# Start both FastAPI applications
CMD ["fastapi", "run", "/src/sequenceServer.py", "--port", "9000", "--host", "0.0.0.0"]