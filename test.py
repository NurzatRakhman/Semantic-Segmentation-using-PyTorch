def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def predict():
    image_transform = ToPILImage()

    predict_transform = Compose([
        Resize((1608, 2022))
    ])
    if predict:
        model = load_checkpoint('checkpoint2.pth')
        test_img = "C:/Users/d068012/Documents/MLExpert/MLExpert/images/rgb.png"
        img = Image.open(test_img).convert('RGB')
        image = input_transform(img)
        height, width = img.size

        output = model(Variable(image, volatile=True).unsqueeze(0))
        label_tf = output[0].data.max(0)[1].unsqueeze(0)

        label = np.transpose(label_tf.cpu().detach().numpy(), (1, 2, 0))
        label_normalized = (((label - label.min()) / (label.max() - label.min())) * 255).astype(np.uint8)
        # label_normalized = (label * 255).astype(np.uint8)
        pil_image = image_transform(label_normalized)
        upsampled_img = predict_transform(pil_image)
        upsampled_img.save('output3.png')


if __name__ == '__main__':
    predict()