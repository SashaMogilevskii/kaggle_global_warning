class AirDataset(Dataset):
    def __init__(self, data, is_train, size):
        self.data = data
        self.size = size
        self.is_train = is_train
        if is_train:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                A.Normalize(),
                ToTensorV2(transpose_mask=True)])


        else:
            self.aug = A.Compose([
                A.Resize(self.size, self.size),
                A.Normalize(),
                ToTensorV2(transpose_mask=True)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path_to_folder = os.path.join(row['head_folder'], str(row['folder']))

        band_8 = np.load(os.path.join(path_to_folder, 'band_8.npy'))
        band_10 = np.load(os.path.join(path_to_folder, 'band_10.npy'))
        band_11 = np.load(os.path.join(path_to_folder, 'band_11.npy'))
        band_12 = np.load(os.path.join(path_to_folder, 'band_12.npy'))
        diff_12_11 = band_12 - band_11
        diff_11_8 = band_11 - band_8

        img = np.clip(np.stack([band_8, band_10, band_11, band_12, diff_11_8, diff_12_11], axis=2), 0, 1)
        img = img[..., 4]
        image = np.load(path_to_img)
        image = image * 255 // 1

        path_to_mask = os.path.join(row['head_folder'], str(row['folder']), 'human_pixel_mask.npy')
        mask = np.load(path_to_mask)

        if self.is_train:
            mask = np.transpose(mask, (1, 2, 0))
            # else:
            #     mask = np.transpose(mask, (2, 0, 1))
            # logger.info(mask.shape)

            data = self.aug(image=image, mask=mask)

            image = data['image']
            mask = data['mask']
            return image, mask
        else:
            data = self.aug(image=image, mask=mask)
            image = data['image']
            standart_mask = np.transpose(mask.copy(), (2, 0, 1))
            mask = data['mask']

            return image, mask, standart_mask