import os
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import weights_init
from utils import transform_config
from data_loader import CIFAR_Paired, DSPRITES_Paired
from networks import Encoder, Decoder
from torch.utils.data import DataLoader
from utils import imshow_grid, mse_loss, reparameterize, l1_loss

import matplotlib.pyplot as plt

def training_procedure(FLAGS):
    """
    model definition
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder.apply(weights_init)

    # load saved models if load_saved flag is true
    if FLAGS.load_saved:
        raise Exception('This is not implemented')
        encoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.encoder_save)))
        decoder.load_state_dict(torch.load(os.path.join('checkpoints', FLAGS.decoder_save)))

    """
    variable definition
    """

    X_1 = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
    X_2 = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)
    X_3 = torch.FloatTensor(FLAGS.batch_size, FLAGS.num_channels, FLAGS.image_size, FLAGS.image_size)

    style_latent_space = torch.FloatTensor(FLAGS.batch_size, FLAGS.style_dim)
    class_latent_space = torch.FloatTensor(FLAGS.batch_size, FLAGS.class_dim)

    """
    loss definitions
    """
    cross_entropy_loss = nn.CrossEntropyLoss()

    '''
    add option to run on GPU
    '''
    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()

        cross_entropy_loss.cuda()

        X_1 = X_1.cuda()
        X_2 = X_2.cuda()
        X_3 = X_3.cuda()

        style_latent_space = style_latent_space.cuda()

    """
    optimizer and scheduler definition
    """
    auto_encoder_optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    reverse_cycle_optimizer = optim.Adam(
        list(encoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    # divide the learning rate by a factor of 10 after 80 epochs
    auto_encoder_scheduler = optim.lr_scheduler.StepLR(auto_encoder_optimizer, step_size=80, gamma=0.1)
    reverse_cycle_scheduler = optim.lr_scheduler.StepLR(reverse_cycle_optimizer, step_size=80, gamma=0.1)

    # Used later to define discriminator ground truths
    Tensor = torch.cuda.FloatTensor if FLAGS.cuda else torch.FloatTensor

    """
    training
    """
    model_root = 'trained_models/%02d_%02d_%02d_%02d/' % (FLAGS.kl_style_coef, FLAGS.kl_class_coef,
                                                          FLAGS.style_dim, FLAGS.class_dim)

    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists(model_root + 'checkpoints'):
        os.makedirs(model_root + 'checkpoints')

    if not os.path.exists(model_root + 'reconstructed_images/'):
        os.makedirs(model_root + 'reconstructed_images/')

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(model_root + FLAGS.log_file, 'w') as log:
            log.write('Epoch\tIteration\tReconstruction_loss\tKL_divergence_loss\tReverse_cycle_loss\n')

    # NOTE not using CIFAR
    # print('Loading CIFAR paired dataset...')
    # paired_cifar = CIFAR_Paired(root='cifar', download=True, train=True, transform=transform_config)
    # loader = cycle(DataLoader(paired_cifar, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    # load data set and create data loader instance
    print('Loading DSPRITES paired dataset...')
    paired_dsprites = DSPRITES_Paired(train=True, transform=transform_config)
    loader = cycle(DataLoader(paired_dsprites, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    # Save a batch of images to use for visualization
    image_sample_1, image_sample_2, _, _ = next(loader)
    image_sample_3, _, _, _ = next(loader)
    # for pair_idx, (img_1, img_2) in enumerate(zip(image_sample_1, image_sample_2)):
    #   plt.imshow(img_1, cmap='gray')
    #   plt.savefig('tmp/%02d_0.png' % pair_idx)
    #   plt.imshow(img_2, cmap='gray')
    #   plt.savefig('tmp/%02d_1.png' % pair_idx)

    # initialize summary writer
    writer = SummaryWriter()

    def kl_loss(logvar, mu, FLAGS):
      loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
      loss /= FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size

      return loss

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '..........................................................................')

        # update the learning rate scheduler
        auto_encoder_scheduler.step()
        reverse_cycle_scheduler.step()

        for iteration in range(int(len(paired_dsprites) / FLAGS.batch_size)):
            # A. run the auto-encoder reconstruction
            image_batch_1, image_batch_2, _, _ = next(loader)

            auto_encoder_optimizer.zero_grad()

            X_1.copy_(image_batch_1)
            X_2.copy_(image_batch_2)

            """ Style and class latent for top graph """
            style_mu_1, style_logvar_1, class_mu_1, class_logvar_1 = encoder(Variable(X_1))
            style_latent_space_1 = reparameterize(training=True, mu=style_mu_1, logvar=style_logvar_1)
            class_latent_space_1 = reparameterize(training=True, mu=class_mu_1, logvar=class_logvar_1)

            # kl_divergence_loss_1 = FLAGS.kl_divergence_coef * (
            #     - 0.5 * torch.sum(1 + style_logvar_1 - style_mu_1.pow(2) - style_logvar_1.exp())
            # )

            kl_style_loss_1 = FLAGS.kl_style_coef * kl_loss(style_logvar_1, style_mu_1, FLAGS)
            kl_style_loss_1.backward(retain_graph=True)

            kl_class_loss_1 = FLAGS.kl_class_coef * kl_loss(class_logvar_1, class_mu_1, FLAGS)
            kl_class_loss_1.backward(retain_graph=True)

            """ Style and class latent for bottom graph """
            style_mu_2, style_logvar_2, class_mu_2, class_logvar_2 = encoder(Variable(X_2))
            style_latent_space_2 = reparameterize(training=True, mu=style_mu_2, logvar=style_logvar_2)
            class_latent_space_2 = reparameterize(training=True, mu=class_mu_2, logvar=class_logvar_2)

            # kl_divergence_loss_2 = FLAGS.kl_divergence_coef * (
            #     - 0.5 * torch.sum(1 + style_logvar_2 - style_mu_2.pow(2) - style_logvar_2.exp())
            # )

            kl_style_loss_2 = FLAGS.kl_style_coef * kl_loss(style_logvar_2, style_mu_2, FLAGS)
            kl_style_loss_2.backward(retain_graph=True)

            kl_class_loss_2 = FLAGS.kl_class_coef * kl_loss(class_logvar_2, class_mu_2, FLAGS)
            kl_class_loss_2.backward(retain_graph=True)

            """ Reconstruction for forward cycle """
            reconstructed_X_1 = decoder(style_latent_space_1, class_latent_space_2)
            reconstructed_X_2 = decoder(style_latent_space_2, class_latent_space_1)

            reconstruction_error_1 = FLAGS.reconstruction_coef * mse_loss(reconstructed_X_1, Variable(X_1))
            reconstruction_error_1.backward(retain_graph=True)

            reconstruction_error_2 = FLAGS.reconstruction_coef * mse_loss(reconstructed_X_2, Variable(X_2))
            reconstruction_error_2.backward()

            reconstruction_error = (reconstruction_error_1 + reconstruction_error_2) / FLAGS.reconstruction_coef
            kl_divergence_error = (kl_style_loss_1 + kl_style_loss_2) / FLAGS.kl_style_coef \
                                    + (kl_class_loss_1 + kl_class_loss_2) / FLAGS.kl_class_coef

            auto_encoder_optimizer.step()

            # B. reverse cycle
            image_batch_1, _, _, _ = next(loader)
            image_batch_2, _, _, _ = next(loader)

            reverse_cycle_optimizer.zero_grad()

            X_1.copy_(image_batch_1)
            X_2.copy_(image_batch_2)

            style_latent_space.normal_(0., 1.)

            _, _, class_mu_1, class_logvar_1 = encoder(Variable(X_1))
            _, _, class_mu_2, class_logvar_2 = encoder(Variable(X_2))
            class_latent_space_1 = reparameterize(training=False, mu=class_mu_1, logvar=class_logvar_1)
            class_latent_space_2 = reparameterize(training=False, mu=class_mu_2, logvar=class_logvar_2)

            reconstructed_X_1 = decoder(Variable(style_latent_space), class_latent_space_1.detach())
            reconstructed_X_2 = decoder(Variable(style_latent_space), class_latent_space_2.detach())

            style_mu_1, style_logvar_1, _, _ = encoder(reconstructed_X_1)
            style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)

            style_mu_2, style_logvar_2, _, _ = encoder(reconstructed_X_2)
            style_latent_space_2 = reparameterize(training=False, mu=style_mu_2, logvar=style_logvar_2)

            reverse_cycle_loss = FLAGS.reverse_cycle_coef * l1_loss(style_latent_space_1, style_latent_space_2)
            reverse_cycle_loss.backward()
            reverse_cycle_loss /= FLAGS.reverse_cycle_coef

            reverse_cycle_optimizer.step()

            if (iteration + 1) % 10 == 0:
                print('')
                print('Epoch #' + str(epoch))
                print('Iteration #' + str(iteration))

                print('')
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('KL-Divergence loss: ' + str(kl_divergence_error.data.storage().tolist()[0]))
                print('Reverse cycle loss: ' + str(reverse_cycle_loss.data.storage().tolist()[0]))

            # write to log
            with open(model_root + FLAGS.log_file, 'a') as log:
                log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                    epoch,
                    iteration,
                    reconstruction_error.data.storage().tolist()[0],
                    kl_divergence_error.data.storage().tolist()[0],
                    reverse_cycle_loss.data.storage().tolist()[0],
                ))

            # write to tensorboard
            writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_dsprites) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('KL-Divergence loss', kl_divergence_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_dsprites) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Reverse cycle loss', reverse_cycle_loss.data.storage().tolist()[0],
                              epoch * (int(len(paired_dsprites) / FLAGS.batch_size) + 1) + iteration)

        # save model after every 5 epochs
        if (epoch + 1) % 1 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(encoder.state_dict(), os.path.join(model_root, 'checkpoints', FLAGS.encoder_save))
            torch.save(decoder.state_dict(), os.path.join(model_root, 'checkpoints', FLAGS.decoder_save))

            """
            save reconstructed images and style swapped image generations to check progress
            """

            X_1.copy_(image_sample_1)
            X_2.copy_(image_sample_2)
            X_3.copy_(image_sample_3)

            style_mu_1, style_logvar_1, _, _ = encoder(Variable(X_1))
            _, _, class_mu_2, class_logvar_2 = encoder(Variable(X_2))
            style_mu_3, style_logvar_3, _, _ = encoder(Variable(X_3))

            style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)
            class_latent_space_2 = reparameterize(training=False, mu=class_mu_2, logvar=class_logvar_2)
            style_latent_space_3 = reparameterize(training=False, mu=style_mu_3, logvar=style_logvar_3)

            reconstructed_X_1_2 = decoder(style_latent_space_1, class_latent_space_2)
            reconstructed_X_3_2 = decoder(style_latent_space_3, class_latent_space_2)

            img_root = model_root + 'reconstructed_images/'

            # save input image batch
            image_batch = np.transpose(X_1.cpu().numpy(), (0, 2, 3, 1))
            image_batch = np.concatenate((image_batch, image_batch, image_batch), axis=3)
            imshow_grid(image_batch, name=img_root + str(epoch) + '_original', save=True)

            # save reconstructed batch
            reconstructed_x = np.transpose(reconstructed_X_1_2.cpu().data.numpy(), (0, 2, 3, 1))
            reconstructed_x = np.concatenate((reconstructed_x, reconstructed_x, reconstructed_x), axis=3)
            imshow_grid(reconstructed_x, name=img_root + str(epoch) + '_target', save=True)

            style_batch = np.transpose(X_3.cpu().numpy(), (0, 2, 3, 1))
            style_batch = np.concatenate((style_batch, style_batch, style_batch), axis=3)
            imshow_grid(style_batch, name=img_root + str(epoch) + '_style', save=True)

            # save style swapped reconstructed batch
            reconstructed_style = np.transpose(reconstructed_X_3_2.cpu().data.numpy(), (0, 2, 3, 1))
            reconstructed_style = np.concatenate((reconstructed_style, reconstructed_style, reconstructed_style), axis=3)
            imshow_grid(reconstructed_style, name=img_root + str(epoch) + '_style_target', save=True)
