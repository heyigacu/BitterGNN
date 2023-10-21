#use Python image as basic image
FROM centos:7

# install dependencies
RUN yum -y update yum groupinstall -y "development tools" && yum install -y make gcc wget gcc-c++ autoconf automake bzip2-devel libffi-devel zlib zlib-devel pcre-devel &&  yum clean all 

###install openssl
RUN wget https://www.openssl.org/source/openssl-1.1.1q.tar.gz --no-check-certificate && tar -zxvf openssl-1.1.1q.tar.gz 
RUN cd openssl-1.1.1q && ./config shared --prefix=/usr/local/openssl && make && make install && cd .. && rm -rf openssl-1.1.1q.tar.gz openssl-1.1.1q
RUN echo "/usr/local/openssl/lib" >> /etc/ld.so.conf && ldconfig && ln -fs /usr/local/openssl/bin/openssl /usr/bin/openssl && ln -s /usr/local/openssl/include/openssl /usr/include/openssl 
RUN  openssl version && sleep 5

###install python3.10.1
RUN wget https://www.python.org/ftp/python/3.10.1/Python-3.10.1.tgz &&  tar xzf Python-3.10.1.tgz && cd Python-3.10.1 && ./configure --enable-optimizations --with-openssl=/usr/local/openssl  && make altinstall && cd .. && rm -rf Python-3.10.1 Python-3.10.1.tgz
RUN python3.10 -V && sleep 5


###install pip
WORKDIR /app
copy requirements.txt  /app
RUN python3.10 -m pip install --upgrade pip &&  pip3.10 install -r requirements.txt  --no-cache-dir
RUN pip3.10 list && sleep 5

# expose poet
#EXPOSE xxx
# define start command
#CMD ["python", "main.py"]
