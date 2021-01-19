FROM python:3.8-slim

WORKDIR /home

COPY *.whl /home/

RUN set -ex \
	\
    && apt-get update && apt-get install -y --no-install-recommends libopenjp2-7 libgl1 \
	&& savedAptMark="$(apt-mark showmanual)" \
	&& apt-get install -y --no-install-recommends gcc g++ \
    \
    && pip install dcimg *.whl \
    && stitch-align -h > /dev/null && stitch-fuse -h > /dev/null \
    && rm -fr *.whl /root/.cache/pip \
    \
    && apt-mark auto '.*' > /dev/null \
	&& apt-mark manual $savedAptMark \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
	&& rm -rf /var/lib/apt/lists/*

CMD ["stitch-align"]
