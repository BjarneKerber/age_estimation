import torch
import torch.nn.functional as F

def gradcam(model, img, prediction, channel=0):
    """Performs Gradcam for a given model and slices
    Args:
        model (model): trained pytorch model
        img (array): CT Slices
        feature_dimension (int, optional): Number of filters for last convolution (Or convolution of interest). Defaults to 512.
    Returns:
        array: GradCam Heatmap
    """
    # Gradient = 0
    model.zero_grad()
    H, W = img.shape[2:]

    mu = torch.squeeze(prediction[:, 0])
    sigma = torch.squeeze(prediction[:, 1])

    # get the gradient of the output with respect to the parameters of the model
    mu.backward(retain_graph=True)

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()
    #     activations = model.get_activations(img[:, channel, :, :]).detach()

    # weight the channels by corresponding gradients
    for j in range(activations.size(dim=1)):
        activations[:, j, :, :] *= pooled_gradients[j]

    heatmap = torch.mean(activations, dim=1)
    heatmap = F.relu(heatmap)
    heatmap -= heatmap.min()
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    return heatmap


def gradcam_pp(model, img, prediction, channel=0):
    """Performs Gradcam++ for a given model and slices
    Args:
        model (model): trained pytorch model
        img (array): CT Slices
    Returns:
        array: GradCam Heatmap
    """
    # Gradient = 0
    model.zero_grad()

    H, W = img.shape[2:]

    mu = torch.squeeze(prediction[:, 0])
    sigma = torch.squeeze(prediction[:, 1])

    # get the gradient of the output with respect to the parameters of the model
    mu.backward(retain_graph=True)

    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    b, k, u, v = gradients.size()

    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()
    #     activations = model.get_activations(img[:, channel, :, :]).detach()

    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + \
                  activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

    alpha = alpha_num.div(alpha_denom + 1e-7)
    positive_gradients = F.relu(mu.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

    saliency_map = (weights * activations).sum(1, keepdim=True)
    saliency_map = F.relu(saliency_map)
    saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
    saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
    #     saliency_map = F.interpolate(saliency_map, (H, W), mode="linear")

    return saliency_map.cpu().numpy()[0][0]


def scorecam(model, img, prediction, channel=0):
    """Computes activation maps by using ScoreCAM
    # https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py
    Args:
        model (torch model): ScoreCAM model
        img (image tensor): Image tensor
        device (device): torch device
    Returns:
        array: ScoreCAM heatmap
    """
    H, W = img.shape[2:]
    model.zero_grad()

    # Gradient = 0
    model.zero_grad()

    mu = torch.squeeze(prediction[:, 0])
    sigma = torch.squeeze(prediction[:, 1])

    # get the gradient of the output with respect to the parameters of the model
    (mu + torch.exp(sigma)).backward(retain_graph=True)

    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()
    #     activations = model.get_activations(img[:, channel, :, :]).detach()

    activations = F.relu(activations)
    activations = F.interpolate(activations, (H, W), mode="bilinear")
    B, C, _, _ = activations.shape

    score_activation = torch.zeros(1, 1, H, W)

    with torch.no_grad():
        for i in range(C):
            activation_k = activations[:, i, :, :]
            normalized_activation_k = (activation_k - activation_k.min()) / (activation_k.max() - activation_k.min())
            pred2 = model(img * normalized_activation_k)[0]
            prob2 = torch.squeeze(pred2)

            score = prob2[0].cpu()
            if torch.isnan(score):
                continue
            score_activation += score * activation_k.cpu()
    score_activation = F.relu(score_activation)

    score_activation_norm = (score_activation - score_activation.min()) / (
                score_activation.max() - score_activation.min())

    return score_activation_norm.cpu().numpy()[0][0]
