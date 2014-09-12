DPTM
=========

DPTM stands for Data Parallel Task Manager. Its objectives are to:

  - Utilize parallel processors with minimum changes to existing serial code
  - Manage and schedule tasks to insure full utilization of available hardware
  - Be compatible with parallel APIs like OpenCL, CUDA, MPSS and more

Specifically, as outlined by Dr. Amir Farbin:

>The goal is to establish infrastructure for hybrid multi-core processor (ie CPU) and many-core co-processor (eg a GPU) computing in complex applications consisting of large numbers ofalgorithms, without major redesign and reworking of legacy software. The vision is that Task Parallel (TP) applications running on CPU cores schedule large numbers of Data Parallel (DP) computing tasks on available many-core co-processors, maximally utilizing the all available hard-ware resources. The project will be executed in the context of offline and trigger reconstruction of Large Hadron Collider (LHC) collisions, but will certainly have a broader impact because it is also applicable to particle interaction simulation (ie Geant4), streaming Data Acquisition Systems, statistical analysis, and other scientific applications.



This text you see here is *actually* written in Markdown! To get a feel for Markdown's syntax, type some text into the left window and watch the results in the right.  

Version
----

2.0

Tech
-----------

Dillinger uses a number of open source projects to work properly:

* [Ace Editor] - awesome web-based text editor
* [Marked] - a super fast port of Markdown to JavaScript
* [Twitter Bootstrap] - great UI boilerplate for modern web apps
* [node.js] - evented I/O for the backend
* [Express] - fast node.js network app framework [@tjholowaychuk]
* [keymaster.js] - awesome keyboard handler lib by [@thomasfuchs]
* [jQuery] - duh 

Installation
--------------

```sh
git clone [git-repo-url] dillinger
cd dillinger
npm i -d
mkdir -p public/files/{md,html,pdf}
```

##### Configure Plugins. Instructions in following README.md files

* plugins/dropbox/README.md
* plugins/github/README.md
* plugins/googledrive/README.md

```sh
node app
```


License
----

MIT


**Free Software, Hell Yeah!**

[john gruber]:http://daringfireball.net/
[@thomasfuchs]:http://twitter.com/thomasfuchs
[1]:http://daringfireball.net/projects/markdown/
[marked]:https://github.com/chjj/marked
[Ace Editor]:http://ace.ajax.org
[node.js]:http://nodejs.org
[Twitter Bootstrap]:http://twitter.github.com/bootstrap/
[keymaster.js]:https://github.com/madrobby/keymaster
[jQuery]:http://jquery.com
[@tjholowaychuk]:http://twitter.com/tjholowaychuk
[express]:http://expressjs.com
