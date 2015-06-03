# -*- mode: ruby -*-
# vi: set ft=ruby :

ipythonPort = 8001                 # Ipython port to forward (also set in IPython notebook config)

Vagrant.configure(2) do |config|
  config.ssh.insert_key = true
  config.vm.define "sparkvm" do |master|
    master.vm.box = "sparkmooc/base"
    master.vm.box_download_insecure = true
    master.vm.boot_timeout = 900
    master.vm.network :forwarded_port, host: ipythonPort, guest: ipythonPort, auto_correct: true   # IPython port (set in notebook config)
    master.vm.network :forwarded_port, host: 4040, guest: 4040, auto_correct: true                 # Spark UI (Driver)
    master.vm.hostname = "sparkvm"
    master.vm.usable_port_range = 4040..4090

    master.vm.provider :virtualbox do |v|
      v.name = master.vm.hostname.to_s
    end
  end
end
