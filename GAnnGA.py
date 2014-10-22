from pyevolve import GSimpleGA, Util, Consts
from GAnnPopulation import GAnnPopulation
from pyfann import libfann
from time  import time
import logging
import code

from sys   import platform as sys_platform


class GAnnGA(GSimpleGA.GSimpleGA):
    def __init__(self, genome, neural_net, train_data, seed=None, interactiveMode=True):
        GSimpleGA.GSimpleGA.__init__(self, genome, seed, interactiveMode)
        if not isinstance(neural_net, libfann.neural_net):
            Util.raiseException("The second argument of the GAnnGA should be an instance of libfann.neural_net",neural_net)
        self.internalPop  = GAnnPopulation(genome)
        self.neuralNet = neural_net
        if not isinstance(train_data, libfann.training_data):
            Util.raiseException("The second argument of the GAnnGA should be an instance of libfann.neural_net",train_data)
        self.trainData = train_data
        self.initializationFlag = False

        
    def setInitializationFlag(self, toTrue=True):
        if not toTrue:
            self.initializationFlag = False
        else:
            self.initializationFlag = True
    def isInitialized(self):
        return self.initializationFlag
    def initialize(self):
      """ Initializes the GA Engine. Create and initialize population """
      self.internalPop.create(minimax=self.minimax)
      self.internalPop.initialize(ga_engine=self)
      self.setInitializationFlag()
      logging.debug("The GA Engine was initialized !")
    def step(self):
        """ Just do one step in evolution, one generation """
        genomeMom = None
        genomeDad = None
  
        newPop = GAnnPopulation(self.internalPop)
        logging.debug("Population was cloned.")
        
        size_iterate = len(self.internalPop)
  
        # Odd population size
        if size_iterate % 2 != 0: size_iterate -= 1
  
        crossover_empty = self.select(popID=self.currentGeneration).crossover.isEmpty()
        
        for i in xrange(0, size_iterate, 2):
           genomeMom = self.select(popID=self.currentGeneration)
           genomeDad = self.select(popID=self.currentGeneration)
           
           if not crossover_empty and self.pCrossover >= 1.0:
              for it in genomeMom.crossover.applyFunctions(mom=genomeMom, dad=genomeDad, count=2):
                 (sister, brother) = it
           else:
              if not crossover_empty and Util.randomFlipCoin(self.pCrossover):
                 for it in genomeMom.crossover.applyFunctions(mom=genomeMom, dad=genomeDad, count=2):
                    (sister, brother) = it
              else:
                 sister = genomeMom.clone()
                 brother = genomeDad.clone()
  
           sister.mutate(pmut=self.pMutation, mutation_nodes=None, ga_engine=self)
           brother.mutate(pmut=self.pMutation, mutation_nodes=None, ga_engine=self)
  
           newPop.internalPop.append(sister)
           newPop.internalPop.append(brother)
  
        if len(self.internalPop) % 2 != 0:
           genomeMom = self.select(popID=self.currentGeneration)
           genomeDad = self.select(popID=self.currentGeneration)
  
           if Util.randomFlipCoin(self.pCrossover):
              for it in genomeMom.crossover.applyFunctions(mom=genomeMom, dad=genomeDad, count=1):
                 (sister, brother) = it
           else:
              sister = random.choice([genomeMom, genomeDad])
              sister = sister.clone()
              sister.mutate(pmut=self.pMutation, ga_engine=self)
  
           newPop.internalPop.append(sister)
  
        logging.debug("Evaluating the new created population.")
        newPop.evaluate(network=self.neuralNet, data=self.trainData)
  
        #Niching methods- Petrowski's clearing
        self.clear()
  
        if self.elitism:
           logging.debug("Doing elitism.")
           if self.getMinimax() == Consts.minimaxType["maximize"]:
              for i in xrange(self.nElitismReplacement):
                 if self.internalPop.bestRaw(i).score > newPop.bestRaw(i).score:
                    newPop[len(newPop)-1-i] = self.internalPop.bestRaw(i)
           elif self.getMinimax() == Consts.minimaxType["minimize"]:
              for i in xrange(self.nElitismReplacement):
                 if self.internalPop.bestRaw(i).score < newPop.bestRaw(i).score:
                    newPop[len(newPop)-1-i] = self.internalPop.bestRaw(i)
  
        self.internalPop = newPop
        self.internalPop.sort()
  
        logging.debug("The generation %d was finished.", self.currentGeneration)
  
        self.currentGeneration += 1
  
        return (self.currentGeneration == self.nGenerations)

    def evolve(self, freq_stats=0):
        stopFlagCallback = False
        stopFlagTerminationCriteria = False
  
        self.time_init = time()
  
        logging.debug("Starting the DB Adapter and the Migration Adapter if any")
        if self.dbAdapter: self.dbAdapter.open(self)
        if self.migrationAdapter: self.migrationAdapter.start()
  
  
        if self.getGPMode():
           gp_function_prefix = self.getParam("gp_function_prefix")
           if gp_function_prefix is not None:
              self.__gp_catch_functions(gp_function_prefix)
  
        if not self.initializationFlag:
            self.initialize()
        self.internalPop.evaluate(network=self.neuralNet,data=self.trainData)
        self.internalPop.sort()
        logging.debug("Starting loop over evolutionary algorithm.")
  
        try:      
           while True:
              if self.migrationAdapter:
                 logging.debug("Migration adapter: exchange")
                 self.migrationAdapter.exchange()
                 self.internalPop.clearFlags()
                 self.internalPop.sort()
  
              if not self.stepCallback.isEmpty():
                 for it in self.stepCallback.applyFunctions(self):
                    stopFlagCallback = it
  
              if not self.terminationCriteria.isEmpty():
                 for it in self.terminationCriteria.applyFunctions(self):
                    stopFlagTerminationCriteria = it
  
              if freq_stats:
                 if (self.currentGeneration % freq_stats == 0) or (self.getCurrentGeneration() == 0):
                    self.printStats()
  
              if self.dbAdapter:
                 if self.currentGeneration % self.dbAdapter.getStatsGenFreq() == 0:
                    self.dumpStatsDB()
  
              if stopFlagTerminationCriteria:
                 logging.debug("Evolution stopped by the Termination Criteria !")
                 if freq_stats:
                    print "\n\tEvolution stopped by Termination Criteria function !\n"
                 break
  
              if stopFlagCallback:
                 logging.debug("Evolution stopped by Step Callback function !")
                 if freq_stats:
                    print "\n\tEvolution stopped by Step Callback function !\n"
                 break
  
              if self.interactiveMode:
                 if sys_platform[:3] == "win":
                    if msvcrt.kbhit():
                       if ord(msvcrt.getch()) == Consts.CDefESCKey:
                          print "Loading modules for Interactive Mode...",
                          logging.debug("Windows Interactive Mode key detected ! generation=%d", self.getCurrentGeneration())
                          from pyevolve import Interaction
                          print " done !"
                          interact_banner = "## Pyevolve v.%s - Interactive Mode ##\nPress CTRL-Z to quit interactive mode." % (pyevolve.__version__,)
                          session_locals = { "ga_engine"  : self,
                                             "population" : self.getPopulation(),
                                             "pyevolve"   : pyevolve,
                                             "it"         : Interaction}
                          print
                          code.interact(interact_banner, local=session_locals)
  
                 if (self.getInteractiveGeneration() >= 0) and (self.getInteractiveGeneration() == self.getCurrentGeneration()):
                          print "Loading modules for Interactive Mode...",
                          logging.debug("Manual Interactive Mode key detected ! generation=%d", self.getCurrentGeneration())
                          from pyevolve import Interaction
                          print " done !"
                          interact_banner = "## Pyevolve v.%s - Interactive Mode ##" % (pyevolve.__version__,)
                          session_locals = { "ga_engine"  : self,
                                             "population" : self.getPopulation(),
                                             "pyevolve"   : pyevolve,
                                             "it"         : Interaction}
                          print
                          code.interact(interact_banner, local=session_locals)
  
              if self.step(): break #exit if the number of generations is equal to the max. number of gens.
  
        except KeyboardInterrupt:
           logging.debug("CTRL-C detected, finishing evolution.")
           if freq_stats: print "\n\tA break was detected, you have interrupted the evolution !\n"
  
        if freq_stats != 0:
           self.printStats()
           self.printTimeElapsed()
  
        if self.dbAdapter:
           logging.debug("Closing the DB Adapter")
           if not (self.currentGeneration % self.dbAdapter.getStatsGenFreq() == 0):
              self.dumpStatsDB()
           self.dbAdapter.commitAndClose()
     
        if self.migrationAdapter:
           logging.debug("Closing the Migration Adapter")
           if freq_stats: print "Stopping the migration adapter... ",
           self.migrationAdapter.stop()
           if freq_stats: print "done !"
  
        return self.bestIndividual()
