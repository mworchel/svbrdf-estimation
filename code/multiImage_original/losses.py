import tensorflow.compat.v1 as tf
import renderer
import helpers


def DX(x):
    return x[:,:,1:,:] - x[:,:,:-1,:]    # so this just subtracts the image from a single-pixel shifted version of itself (while cropping out two pixels because we don't know what's outside the image)

def DY(x):
    return x[:,1:,:,:] - x[:,:-1,:,:]    # likewise for y-direction


#Compute the average L1 difference of two tensors
def l1(x, y):
    return tf.reduce_mean(tf.abs(x-y))

#Compute the average L2 difference of two tensors
def l2(x, y):
    return tf.reduce_mean(tf.square(x-y))

#Epsilons so that the logs don't end up with log(0) as it's undefined
epsilonL1 = 0.01
epsilonRender = 0.1
epsilonL2 = 0.1

#Defines the loss object with all the parameters we will need
class Loss:
    lossType = "render"
    batchSize = 8
    lossValue = 0
    crop_size = 256
    lr = 0.00002
    beta1Adam = 0.5
    
    #This defines the number of renderings for the rendering loss at each step.
    nbDiffuseRendering = 3
    nbSpecularRendering = 6
    
    #Include the diffuse part of the rendering or not in the specular renderings    
    includeDiffuse = False

    outputs = None
    targets = None
    surfaceArray = None
    outputsRenderings = None
    targetsRenderings = None
    trainOp =  None

    def __init__(self, lossType, outputs, targets, crop_size, batchSize, lr, includeDiffuse) :
        self.lossType = lossType
        self.outputs = outputs
        self.targets = targets
        self.crop_size = crop_size
        self.batchSize = batchSize
        self.lr = lr
        self.includeDiffuse = includeDiffuse

    #Compute the loss using the gradient of the parameters.
    def __loss_grad(self, alpha=0.2):
        loss_val = alpha * self.__l1Loss()#l1(self.outputs, self.targets) # here alpha is a weighting for the direct pixel value comparison, you can make it surprisingly low, though it's possible that 0.1 might be too low for some problems
        loss_val = loss_val + l1(DX(self.outputs), DX(self.targets))
        loss_val = loss_val + l1(DY(self.outputs), DY(self.targets))
        return loss_val

    #Compute the loss using the L2 on the parameters.
    def __l2Loss(self):
        #outputs have shape [?, height, width, 12]
        #targets have shape [?, height, width, 12]
        outputsNormal = self.outputs[:,:,:,0:3]
        outputsDiffuse = tf.log(epsilonL2 + helpers.deprocess(self.outputs[:,:,:,3:6]))
        outputsRoughness = self.outputs[:,:,:,6:9]
        outputsSpecular = tf.log(epsilonL2 + helpers.deprocess(self.outputs[:,:,:,9:12]))

        targetsNormal = self.targets[:,:,:,0:3]
        targetsDiffuse = tf.log(epsilonL2 + helpers.deprocess(self.targets[:,:,:,3:6]))
        targetsRoughness = self.targets[:,:,:,6:9]
        targetsSpecular = tf.log(epsilonL2 + helpers.deprocess(self.targets[:,:,:,9:12]))

        return l2(outputsNormal, targetsNormal) + l2(outputsDiffuse, targetsDiffuse) + l2(outputsRoughness, targetsRoughness) + l2(outputsSpecular, targetsSpecular)
    
    #Compute the loss using the L1 on the parameters.
    def __l1Loss(self):
        #outputs have shape [?, height, width, 12]
        #targets have shape [?, height, width, 12]
        outputsNormal = self.outputs[:,:,:,0:3]
        outputsDiffuse = tf.log(epsilonL1 + helpers.deprocess(self.outputs[:,:,:,3:6]))
        outputsRoughness = self.outputs[:,:,:,6:9]
        outputsSpecular = tf.log(epsilonL1 + helpers.deprocess(self.outputs[:,:,:,9:12]))

        targetsNormal = self.targets[:,:,:,0:3]
        targetsDiffuse = tf.log(epsilonL1 + helpers.deprocess(self.targets[:,:,:,3:6]))
        targetsRoughness = self.targets[:,:,:,6:9]
        targetsSpecular = tf.log(epsilonL1 + helpers.deprocess(self.targets[:,:,:,9:12]))

        return l1(outputsNormal, targetsNormal) + l1(outputsDiffuse, targetsDiffuse) + l1(outputsRoughness, targetsRoughness) + l1(outputsSpecular, targetsSpecular)
    
    #Generate the different renderings and concatenate the diffuse an specular renderings.
    def __generateRenderings(self, renderer):
        diffuses = helpers.tf_generateDiffuseRendering(self.batchSize, self.nbDiffuseRendering, self.targets, self.outputs, renderer)
        speculars = helpers.tf_generateSpecularRendering(self.batchSize, self.nbSpecularRendering, self.surfaceArray, self.targets, self.outputs, renderer)
        targetsRendered = tf.concat([diffuses[0],speculars[0]], axis = 1)
        outputsRendered = tf.concat([diffuses[1],speculars[1]], axis = 1)
        return targetsRendered, outputsRendered
    
    #Compute the rendering loss
    def __renderLoss(self):
        #Generate the grid of position between -1,1 used for rendering
        self.surfaceArray = helpers.generateSurfaceArray(self.crop_size)
        
        #Initialize the renderer
        rendererImpl = renderer.GGXRenderer(includeDiffuse = self.includeDiffuse)
        self.targetsRenderings, self.outputsRenderings = self.__generateRenderings(rendererImpl)
        
        #Compute the L1 loss between the renderings, epsilon are here to avoid log(0)        
        targetsLogged = tf.log(self.targetsRenderings + epsilonRender) # Bias could be improved to 0.1 if the network gets upset.
        outputsLogged = tf.log(self.outputsRenderings + epsilonRender)
        lossTotal = l1(targetsLogged, outputsLogged)#0.5 * l1(targetsLogged, outputsLogged) + l1(DX(targetsLogged), DX(outputsLogged)) + l1(DY(targetsLogged), DY(outputsLogged))
       
        #Return the loss
        return lossTotal
        
    #Compute both the rendering loss and the L1 loss on parameters and return this value.
    def __mixedLoss(self, lossL1Factor = 0.1):
        return self.__renderLoss() + lossL1Factor * self.__l1Loss() #self.__loss_grad()

    #Create the loss graph depending on which option was chosen by the user
    def createLossGraph(self):
        if self.lossType == "render":
            self.lossValue = self.__renderLoss()
        elif self.lossType == "l1":
            self.lossValue = self.__l1Loss()
        elif self.lossType == "mixed":
            self.lossValue = self.__mixedLoss()
        else:
            raise ValueError('No such loss: ' + self.lossType)

    #In this function we create the training operations
    def createTrainVariablesGraph(self, reuse_bool = False):
        global_step = tf.train.get_or_create_global_step()
        #Save the current learning rate value (as it ramps up)
        tf.summary.scalar("lr", self.lr)
        with tf.name_scope("model_train"):
            with tf.variable_scope("model_train0", reuse=reuse_bool):
                #Get all the variables which name start with trainableModel (defined in model.py)            
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("trainableModel/")]
                #Initialize Adam optimizer                
                gen_optim = tf.train.AdamOptimizer(self.lr, self.beta1Adam)
                #Compute the gradient required to get the loss using the variables starting with trainableModel                
                gen_grads_and_vars = gen_optim.compute_gradients(self.lossValue, var_list=gen_tvars)
                #Define the training as applying the gradient on the variables starting with trainModel                
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        #Use an EMA to have more smooth loss
        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([self.lossValue])
        self.lossValue = ema.average(self.lossValue)
        incr_global_step = tf.assign(global_step, global_step+1)
        #Define the training step operation as the update of the loss, add 1 to the global step, and train the variable.        
        self.trainOp = tf.group(update_losses, incr_global_step, gen_train)

