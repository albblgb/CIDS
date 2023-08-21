import torch
torch.set_num_threads(8)
import torch.nn as nn
import os
from tqdm import tqdm
import logging
import numpy as np
import numpy as np
import torch
import os
from torchvision.utils import save_image
from tqdm.contrib import tzip

from models import Model
from utils import MetricMonitor, cat_map, logger_info, calculate_psnr, calculate_ssim, calculate_mae, calculate_rmse, calculate_psnrs, calculate_ssims, calculate_maes, calculate_rmses, load_dataset, mkdirs
import config as c


os.environ["CUDA_VISIBLE_DEVICES"] = c.cids_device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_save_path = os.path.join(c.model_dir, 'cids')
mkdirs(model_save_path)

mkdirs('results')
logger_name = 'cids_trained_on_div2k'
logger_info(logger_name, log_path=os.path.join('results', logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info('#'*50)
logger.info('model: CIDSNet')
logger.info('train data path: {:s}'.format(c.trainset_path))
logger.info('test data path: {:s}'.format(c.test_secret_image_path))
logger.info('mode: {:s}'.format(c.mode))

train_loader, test_secret_loader, test_cover_loader = load_dataset(c.trainset_path, c.test_secret_image_path, c.cids_batch_size_train, c.cids_batch_size_test, c.test_cover_image_path,)

model = Model().to(device)

if c.mode == 'test':

    model.load_state_dict(torch.load(c.test_cids_path))

    with torch.no_grad():
        S_psnr = []; S_ssim = []; S_mae = []; S_rmse = []
        R_psnr = []; R_ssim = []; R_mae = []; R_rmse = []
    
        model.eval()
        stream = tqdm(tzip(test_secret_loader, test_cover_loader))
        for idx, (secret, cover) in enumerate(stream):
            # data = data.cuda()
            secret = secret.to(device)
            cover = cover.to(device)

            if c.obfuscate_secret_image == True:
                secret_obf = cat_map(secret.cpu(), obfuscate=True).to(device)

                ################## forward ####################
                stego, obfuscated_resi, stego_resi, secret_rev = model(secret_obf, cover)

                secret_rev = cat_map(secret_rev.cpu(), obfuscate=False).cuda()
            else:    
                # ################## forward ####################
                stego, obfuscated_resi, stego_resi, secret_rev = model(secret, cover)


            # ################## forward ####################
            stego, obfuscated_resi, stego_resi, secret_rev = model(secret, cover)

            stego_resi_obf = abs(obfuscated_resi) * 15
            stego_resi = abs(stego_resi) * 15
            secret_resi = abs(secret - secret_rev) * 15

            if c.save_processed_img == True:
                super_dirs = ['cover', 'secret', 'stego', 'secret_rev', 'stego_resi', 'stego_resi_obf', 'secret_resi']
                for cur_dir in super_dirs:
                    mkdirs(os.path.join(c.img_save_dir, 'cids', c.testset_name, cur_dir))    
                batch_size = cover.shape[0]
                for i in range(batch_size):   
                    image_name = '%.4d.' % (idx * batch_size + i) + c.suffix
                    save_image(cover[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[0], image_name))
                    save_image(secret[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[1], image_name))
                    save_image(stego[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[2], image_name))
                    save_image(secret_rev[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[3], image_name))
                    save_image(stego_resi[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[4], image_name))
                    save_image(stego_resi_obf[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[5], image_name))
                    save_image(secret_resi[i], os.path.join(c.img_save_dir, 'cids', c.testset_name, super_dirs[6], image_name))


            secret_rev = secret_rev.detach().cpu().numpy() * 255
            np.clip(secret_rev, 0, 255)
            secret = secret.detach().cpu().numpy() * 255
            np.clip(secret, 0, 255)
            cover = cover.detach().cpu().numpy() * 255
            np.clip(cover, 0, 255)
            stego = stego.detach().cpu().numpy() * 255
            np.clip(stego, 0, 255)
            
            psnr_temp = calculate_psnrs(cover, stego)
            S_psnr.append(psnr_temp)
            psnr_temp = calculate_psnrs(secret, secret_rev)
            R_psnr.append(psnr_temp)

            mae_temp = calculate_maes(cover, stego)
            S_mae.append(mae_temp)
            mae_temp = calculate_maes(secret, secret_rev)
            R_mae.append(mae_temp)

            rmse_temp = calculate_rmses(cover, stego)
            S_rmse.append(rmse_temp)
            rmse_temp = calculate_rmses(secret, secret_rev)
            R_rmse.append(rmse_temp)

            ssim_temp = calculate_ssims(cover, stego)
            S_ssim.append(ssim_temp)
            ssim_temp = calculate_ssims(secret, secret_rev)
            R_ssim.append(ssim_temp)


        logger.info('testing, stego_avg_psnr: {:.2f}, secref_avg_psnr: {:.2f}'.format(np.mean(S_psnr), np.mean(R_psnr)))
        logger.info('testing, stego_avg_ssim: {:.4f}, secref_avg_ssim: {:.4f}'.format(np.mean(S_ssim), np.mean(R_ssim)))
        logger.info('testing, stego_avg_mae: {:.2f}, secref_avg_mae: {:.2f}'.format(np.mean(S_mae), np.mean(R_mae)))
        logger.info('testing, stego_avg_rmse: {:.2f}, secref_avg_rmse: {:.2f}'.format(np.mean(S_rmse), np.mean(R_rmse)))
else:
    secret_restruction_loss = nn.MSELoss().cuda()
    stego_similarity_loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.weight_step, gamma=c.gamma)

    for epoch in range(c.epochs):
        epoch += 1
        s_loss = []
        r_loss = []
        loss_history=[]
        ###############################################################
        #                            train                            # 
        ###############################################################
        model.train()
        metric_monitor = MetricMonitor(float_precision=5)
        stream = tqdm(train_loader)

        for batch_idx, data in enumerate(stream):
            data = data.cuda()
            secret = data

            cover = (torch.ones_like(secret)*(128/255)).to(device) # gray cover images
            
            ################## forward ####################
            stego, stego_resi, secret_rev = model(secret, cover)

            ################### loss ######################
            S_loss = stego_similarity_loss(cover, stego)
            R_loss = secret_restruction_loss(secret, secret_rev)
            loss =   S_loss + c.cids_beta * R_loss

            ################### backword ##################
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            s_loss.append(S_loss.item())
            r_loss.append(R_loss.item())
            loss_history.append(loss.item())

            metric_monitor.update("S_loss", np.mean(np.array(s_loss)))
            metric_monitor.update("R_loss", np.mean(np.array(r_loss)))
            metric_monitor.update("T_Loss", np.mean(np.array(loss_history)))
            stream.set_description(
                "Epoch: {epoch}. Train.   {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        epoch_losses = np.mean(np.array(loss_history))

        ###############################################################
        #                              val                            # 
        ###############################################################
        model.eval()
        if epoch % c.test_freq == 0:
            with torch.no_grad():
                S_psnr = []
                R_psnr = []
                for data in test_secret_loader:
                    data = data.cuda()
                    secret = data

                    # secret = data[data.shape[0]//2:]
                    # cover = data[:data.shape[0]//2]
                    cover = (torch.ones_like(secret)*(128/255)).to(device)

                    ################## forward ####################
                    stego, stego_resi, secret_rev = model(secret, cover)

                    ############### calculate psnr #################
                    secret_rev = secret_rev.detach().cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.detach().cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.detach().cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    stego = stego.detach().cpu().numpy().squeeze() * 255
                    np.clip(stego, 0, 255)
                    psnr_temp = calculate_psnr(cover, stego)
                    S_psnr.append(psnr_temp)
                    psnr_temp = calculate_psnr(secret, secret_rev)
                    R_psnr.append(psnr_temp)
                logger.info('epoch: {}, training,  loss: {}'.format(epoch, epoch_losses))
                logger.info('epoch: {}, testing, stego_avg_psnr: {:.2f}, secref_avg_psnr: {:.2f}'.format(epoch, np.mean(S_psnr), np.mean(R_psnr)))

        if epoch % c.save_freq == 0 and epoch >= c.save_start_epoch:
            torch.save(model.state_dict(), os.path.join(model_save_path, 'checkpoint_%.4i' % epoch + '.pt'))
            
        scheduler.step()


    
