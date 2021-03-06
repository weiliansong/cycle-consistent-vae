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
from data_loader import CIFAR_Paired
from networks import Encoder, Decoder, Discriminator
from torch.utils.data import DataLoader
from utils import imshow_grid, mse_loss, reparameterize, l1_loss


def training_procedure(FLAGS):
    """
    model definition
    """
    encoder = Encoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    encoder.apply(weights_init)

    decoder = Decoder(style_dim=FLAGS.style_dim, class_dim=FLAGS.class_dim)
    decoder.apply(weights_init)

    discriminator = Discriminator()
    discriminator.apply(weights_init)

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

    """
    loss definitions
    """
    cross_entropy_loss = nn.CrossEntropyLoss()
    adversarial_loss = nn.BCELoss()

    '''
    add option to run on GPU
    '''
    if FLAGS.cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()

        cross_entropy_loss.cuda()
        adversarial_loss.cuda()

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

    generator_optimizer = optim.Adam(
        list(decoder.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    discriminator_optimizer = optim.Adam(
        list(discriminator.parameters()),
        lr=FLAGS.initial_learning_rate,
        betas=(FLAGS.beta_1, FLAGS.beta_2)
    )

    # divide the learning rate by a factor of 10 after 80 epochs
    auto_encoder_scheduler = optim.lr_scheduler.StepLR(auto_encoder_optimizer, step_size=80, gamma=0.1)
    reverse_cycle_scheduler = optim.lr_scheduler.StepLR(reverse_cycle_optimizer, step_size=80, gamma=0.1)
    generator_scheduler = optim.lr_scheduler.StepLR(generator_optimizer, step_size=80, gamma=0.1)
    discriminator_scheduler = optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=80, gamma=0.1)

    # Used later to define discriminator ground truths
    Tensor = torch.cuda.FloatTensor if FLAGS.cuda else torch.FloatTensor

    """
    training
    """
    if torch.cuda.is_available() and not FLAGS.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if not os.path.exists('reconstructed_images'):
        os.makedirs('reconstructed_images')

    # load_saved is false when training is started from 0th iteration
    if not FLAGS.load_saved:
        with open(FLAGS.log_file, 'w') as log:
            headers = ['Epoch', 'Iteration', 'Reconstruction_loss', 'KL_divergence_loss', 'Reverse_cycle_loss']

            if FLAGS.forward_gan:
              headers.extend(['Generator_forward_loss', 'Discriminator_forward_loss'])

            if FLAGS.reverse_gan:
              headers.extend(['Generator_reverse_loss', 'Discriminator_reverse_loss'])

            log.write('\t'.join(headers) + '\n')

    # load data set and create data loader instance
    print('Loading CIFAR paired dataset...')
    paired_cifar = CIFAR_Paired(root='cifar', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(paired_cifar, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0, drop_last=True))

    # Save a batch of images to use for visualization
    image_sample_1, image_sample_2, _ = next(loader)
    image_sample_3, _, _ = next(loader)

    # initialize summary writer
    writer = SummaryWriter()

    for epoch in range(FLAGS.start_epoch, FLAGS.end_epoch):
        print('')
        print('Epoch #' + str(epoch) + '..........................................................................')

        # update the learning rate scheduler
        auto_encoder_scheduler.step()
        reverse_cycle_scheduler.step()
        generator_scheduler.step()
        discriminator_scheduler.step()

        for iteration in range(int(len(paired_cifar) / FLAGS.batch_size)):
            # Adversarial ground truths
            valid = Variable(Tensor(FLAGS.batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(FLAGS.batch_size, 1).fill_(0.0), requires_grad=False)

            # A. run the auto-encoder reconstruction
            image_batch_1, image_batch_2, _ = next(loader)

            auto_encoder_optimizer.zero_grad()

            X_1.copy_(image_batch_1)
            X_2.copy_(image_batch_2)

            style_mu_1, style_logvar_1, class_latent_space_1 = encoder(Variable(X_1))
            style_latent_space_1 = reparameterize(training=True, mu=style_mu_1, logvar=style_logvar_1)

            kl_divergence_loss_1 = FLAGS.kl_divergence_coef * (
                - 0.5 * torch.sum(1 + style_logvar_1 - style_mu_1.pow(2) - style_logvar_1.exp())
            )
            kl_divergence_loss_1 /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
            kl_divergence_loss_1.backward(retain_graph=True)

            style_mu_2, style_logvar_2, class_latent_space_2 = encoder(Variable(X_2))
            style_latent_space_2 = reparameterize(training=True, mu=style_mu_2, logvar=style_logvar_2)

            kl_divergence_loss_2 = FLAGS.kl_divergence_coef * (
                - 0.5 * torch.sum(1 + style_logvar_2 - style_mu_2.pow(2) - style_logvar_2.exp())
            )
            kl_divergence_loss_2 /= (FLAGS.batch_size * FLAGS.num_channels * FLAGS.image_size * FLAGS.image_size)
            kl_divergence_loss_2.backward(retain_graph=True)

            reconstructed_X_1 = decoder(style_latent_space_1, class_latent_space_2)
            reconstructed_X_2 = decoder(style_latent_space_2, class_latent_space_1)

            reconstruction_error_1 = FLAGS.reconstruction_coef * mse_loss(reconstructed_X_1, Variable(X_1))
            reconstruction_error_1.backward(retain_graph=True)

            reconstruction_error_2 = FLAGS.reconstruction_coef * mse_loss(reconstructed_X_2, Variable(X_2))
            reconstruction_error_2.backward()

            reconstruction_error = (reconstruction_error_1 + reconstruction_error_2) / FLAGS.reconstruction_coef
            kl_divergence_error = (kl_divergence_loss_1 + kl_divergence_loss_2) / FLAGS.kl_divergence_coef

            auto_encoder_optimizer.step()

            # A-1. Discriminator training during forward cycle
            if FLAGS.forward_gan:
              # Training generator
              generator_optimizer.zero_grad()

              g_loss_1 = adversarial_loss(discriminator(Variable(reconstructed_X_1)), valid)
              g_loss_2 = adversarial_loss(discriminator(Variable(reconstructed_X_2)), valid)

              gen_f_loss = (g_loss_1 + g_loss_2) / 2.0
              gen_f_loss.backward()

              generator_optimizer.step()

              # Training discriminator
              discriminator_optimizer.zero_grad()

              real_loss_1 = adversarial_loss(discriminator(Variable(X_1)), valid)
              real_loss_2 = adversarial_loss(discriminator(Variable(X_2)), valid)
              fake_loss_1 = adversarial_loss(discriminator(Variable(reconstructed_X_1)), fake)
              fake_loss_2 = adversarial_loss(discriminator(Variable(reconstructed_X_2)), fake)

              dis_f_loss = (real_loss_1 + real_loss_2 + fake_loss_1 + fake_loss_2) / 4.0
              dis_f_loss.backward()

              discriminator_optimizer.step()

            # B. reverse cycle
            image_batch_1, _, __ = next(loader)
            image_batch_2, _, __ = next(loader)

            reverse_cycle_optimizer.zero_grad()

            X_1.copy_(image_batch_1)
            X_2.copy_(image_batch_2)

            style_latent_space.normal_(0., 1.)

            _, __, class_latent_space_1 = encoder(Variable(X_1))
            _, __, class_latent_space_2 = encoder(Variable(X_2))

            reconstructed_X_1 = decoder(Variable(style_latent_space), class_latent_space_1.detach())
            reconstructed_X_2 = decoder(Variable(style_latent_space), class_latent_space_2.detach())

            style_mu_1, style_logvar_1, _ = encoder(reconstructed_X_1)
            style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)

            style_mu_2, style_logvar_2, _ = encoder(reconstructed_X_2)
            style_latent_space_2 = reparameterize(training=False, mu=style_mu_2, logvar=style_logvar_2)

            reverse_cycle_loss = FLAGS.reverse_cycle_coef * l1_loss(style_latent_space_1, style_latent_space_2)
            reverse_cycle_loss.backward()
            reverse_cycle_loss /= FLAGS.reverse_cycle_coef

            reverse_cycle_optimizer.step()

            # B-1. Discriminator training during reverse cycle
            if FLAGS.reverse_gan:
              # Training generator
              generator_optimizer.zero_grad()

              g_loss_1 = adversarial_loss(discriminator(Variable(reconstructed_X_1)), valid)
              g_loss_2 = adversarial_loss(discriminator(Variable(reconstructed_X_2)), valid)

              gen_r_loss = (g_loss_1 + g_loss_2) / 2.0
              gen_r_loss.backward()

              generator_optimizer.step()

              # Training discriminator
              discriminator_optimizer.zero_grad()

              real_loss_1 = adversarial_loss(discriminator(Variable(X_1)), valid)
              real_loss_2 = adversarial_loss(discriminator(Variable(X_2)), valid)
              fake_loss_1 = adversarial_loss(discriminator(Variable(reconstructed_X_1)), fake)
              fake_loss_2 = adversarial_loss(discriminator(Variable(reconstructed_X_2)), fake)

              dis_r_loss = (real_loss_1 + real_loss_2 + fake_loss_1 + fake_loss_2) / 4.0
              dis_r_loss.backward()

              discriminator_optimizer.step()

            if (iteration + 1) % 10 == 0:
                print('')
                print('Epoch #' + str(epoch))
                print('Iteration #' + str(iteration))

                print('')
                print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
                print('KL-Divergence loss: ' + str(kl_divergence_error.data.storage().tolist()[0]))
                print('Reverse cycle loss: ' + str(reverse_cycle_loss.data.storage().tolist()[0]))

                if FLAGS.forward_gan:
                  print('Generator F loss: ' + str(gen_f_loss.data.storage().tolist()[0]))
                  print('Discriminator F loss: ' + str(dis_f_loss.data.storage().tolist()[0]))

                if FLAGS.reverse_gan:
                  print('Generator R loss: ' + str(gen_r_loss.data.storage().tolist()[0]))
                  print('Discriminator R loss: ' + str(dis_r_loss.data.storage().tolist()[0]))

            # write to log
            with open(FLAGS.log_file, 'a') as log:
                row = []

                row.append(epoch)
                row.append(iteration)
                row.append(reconstruction_error.data.storage().tolist()[0])
                row.append(kl_divergence_error.data.storage().tolist()[0])
                row.append(reverse_cycle_loss.data.storage().tolist()[0])

                if FLAGS.forward_gan:
                  row.append(gen_f_loss.data.storage().tolist()[0])
                  row.append(dis_f_loss.data.storage().tolist()[0])

                if FLAGS.reverse_gan:
                  row.append(gen_r_loss.data.storage().tolist()[0])
                  row.append(dis_r_loss.data.storage().tolist()[0])

                row = [str(x) for x in row]
                log.write('\t'.join(row) + '\n')

            # write to tensorboard
            writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('KL-Divergence loss', kl_divergence_error.data.storage().tolist()[0],
                              epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)
            writer.add_scalar('Reverse cycle loss', reverse_cycle_loss.data.storage().tolist()[0],
                              epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)

            if FLAGS.forward_gan:
              writer.add_scalar('Generator F loss', gen_f_loss.data.storage().tolist()[0],
                                epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)
              writer.add_scalar('Discriminator F loss', dis_f_loss.data.storage().tolist()[0],
                                epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)

            if FLAGS.reverse_gan:
              writer.add_scalar('Generator R loss', gen_r_loss.data.storage().tolist()[0],
                                epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)
              writer.add_scalar('Discriminator R loss', dis_r_loss.data.storage().tolist()[0],
                                epoch * (int(len(paired_cifar) / FLAGS.batch_size) + 1) + iteration)

        # save model after every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.end_epoch:
            torch.save(encoder.state_dict(), os.path.join('checkpoints', FLAGS.encoder_save))
            torch.save(decoder.state_dict(), os.path.join('checkpoints', FLAGS.decoder_save))

            """
            save reconstructed images and style swapped image generations to check progress
            """

            X_1.copy_(image_sample_1)
            X_2.copy_(image_sample_2)
            X_3.copy_(image_sample_3)

            style_mu_1, style_logvar_1, _ = encoder(Variable(X_1))
            _, __, class_latent_space_2 = encoder(Variable(X_2))
            style_mu_3, style_logvar_3, _ = encoder(Variable(X_3))

            style_latent_space_1 = reparameterize(training=False, mu=style_mu_1, logvar=style_logvar_1)
            style_latent_space_3 = reparameterize(training=False, mu=style_mu_3, logvar=style_logvar_3)

            reconstructed_X_1_2 = decoder(style_latent_space_1, class_latent_space_2)
            reconstructed_X_3_2 = decoder(style_latent_space_3, class_latent_space_2)

            # save input image batch
            image_batch = np.transpose(X_1.cpu().numpy(), (0, 2, 3, 1))
            if FLAGS.num_channels == 1:
              image_batch = np.concatenate((image_batch, image_batch, image_batch), axis=3)
            imshow_grid(image_batch, name=str(epoch) + '_original', save=True)

            # save reconstructed batch
            reconstructed_x = np.transpose(reconstructed_X_1_2.cpu().data.numpy(), (0, 2, 3, 1))
            if FLAGS.num_channels == 1:
              reconstructed_x = np.concatenate((reconstructed_x, reconstructed_x, reconstructed_x), axis=3)
            imshow_grid(reconstructed_x, name=str(epoch) + '_target', save=True)

            style_batch = np.transpose(X_3.cpu().numpy(), (0, 2, 3, 1))
            if FLAGS.num_channels == 1:
              style_batch = np.concatenate((style_batch, style_batch, style_batch), axis=3)
            imshow_grid(style_batch, name=str(epoch) + '_style', save=True)

            # save style swapped reconstructed batch
            reconstructed_style = np.transpose(reconstructed_X_3_2.cpu().data.numpy(), (0, 2, 3, 1))
            if FLAGS.num_channels == 1:
              reconstructed_style = np.concatenate((reconstructed_style, reconstructed_style, reconstructed_style), axis=3)
            imshow_grid(reconstructed_style, name=str(epoch) + '_style_target', save=True)
