% Load the training data
load('TrainingSamplesDCT_8_new.mat');

% Compute the histogram estimate of the prior for cheetah
num_cheetah = size(TrainsampleDCT_FG,1);
num_grass = size(TrainsampleDCT_BG,1);
total_samples = num_cheetah + num_grass;
prior_cheetah = num_cheetah / total_samples;

% Compute the histogram estimate of the prior for grass
prior_grass = num_grass / total_samples;

% Compute the mean and variance all X's for the foreground (cheetah) class
mean_FG = mean(TrainsampleDCT_FG);
var_FG = var(TrainsampleDCT_FG);

% Compute the mean and variance all X's for the foreground (cheetah) class
mean_BG = mean(TrainsampleDCT_BG);
var_BG = var(TrainsampleDCT_BG);


mu_cheetah = (mean_FG);
sigma_cheetah = (var_FG);
mu_grass = (mean_BG);
sigma_grass = (var_BG);

% Create a figure with 64 subplots
figure;
for i = 1:64
    x = linspace(min(mu_cheetah(i)- 3*sqrt(sigma_cheetah(i)),mu_grass(i)-3*sqrt(sigma_grass(i))),max(mu_cheetah(i)+3*sqrt(sigma_cheetah(i)),mu_grass(i)+3*sqrt(sigma_grass(i))),100);

    % Compute the marginal densities for cheetah and grass
    y_cheetah = normpdf(x, mu_cheetah(i), sqrt(sigma_cheetah(i)));
    y_grass = normpdf(x, mu_grass(i), sqrt(sigma_grass(i)));

    % Plot the marginal densities with different line styles
    subplot(8,8,i)
    plot(x, y_cheetah, x, y_grass)
    xlabel(sprintf('Feature %d', i));
    ylabel('Density');
end


best = [1, 2, 3, 4, 5, 6, 8, 26];

% Create a figure with 64 subplots
figure;
for i = 1:8
    x = linspace(min(mu_cheetah(best(i))- 3*sqrt(sigma_cheetah(best(i))),mu_grass(best(i))-3*sqrt(sigma_grass(best(i)))),max(mu_cheetah(i)+3*sqrt(sigma_cheetah(i)),mu_grass(i)+3*sqrt(sigma_grass(i))),100);

    % Compute the marginal densities for cheetah and grass
    y_cheetah = normpdf(x, mu_cheetah(best(i)), sqrt(sigma_cheetah(best(i))));
    y_grass = normpdf(x, mu_grass(best(i)), sqrt(sigma_grass(best(i))));

    % Plot the marginal densities with different line styles
    subplot(2,4,i)
    plot(x, y_cheetah, x, y_grass)
    xlabel(sprintf('Feature %d', best(i)));
    ylabel('Density');
end


worst = [63, 37, 55, 51, 48, 56, 64, 35];

% Create a figure with 64 subplots
figure;
for i = 1:8
    x = linspace(-0.05,0.05,100);

    % Compute the marginal densities for cheetah and grass
    y_cheetah = normpdf(x, mu_cheetah(worst(i)), sqrt(sigma_cheetah(worst(i))));
    y_grass = normpdf(x, mu_grass(worst(i)), sqrt(sigma_grass(worst(i))));

    % Plot the marginal densities with different line styles
    subplot(2,4,i)
    plot(x, y_cheetah, x, y_grass)
    xlabel(sprintf('Feature %d', worst(i)));
    ylabel('Density');
end

simg = imread('cheetah.bmp');
dimg = im2double(simg);
mask = im2double(imread('cheetah_mask.bmp'));
error = 0;

% Pad the image with 4 rows of zeros
img = padarray(dimg, [0 0], 0, 'post');

% Define the traversal order
order = [0   1   5   6  14  15  27  28;
         2   4   7  13  16  26  29  42;
         3   8  12  17  25  30  41  43;
         9  11  18  24  31  40  44  53;
       	10  19  23  32  39  45  52  54;
       	20  22  33  38  46  51  55  60;
       	21  34  37  47  50  56  59  61;
       	35  36  48  49  57  58  62  63];

% Initialize the vector
vector = zeros(1, 64);

cov_c = cov(TrainsampleDCT_FG);
cov_g = cov(TrainsampleDCT_BG);
inv_cov_c = inv(cov_c);
inv_cov_g = inv(cov_g);

% Compute the state variable Y for each block in the image
A = zeros((size(img,1)), (size(img,2)));

for i = 4:size(img,1)-4
    for j = 4:size(img,2)-4
        % Extract an 8x8 block from the image
        block = img((i-4)+1:i+4, (j-4)+1:j+4);
        
        B = dct2(block);
        
        % Traverse the matrix in the given order and fill the vector
        for k = 0:numel(B)-1
            [row, col] = find(order==k);
            vector(k+1) = B(row, col);
        end

        % Compute the feature X (index of DCT coefficient with second-largest energy)
       
        prob_cheetah = (vector - mean_FG)*inv_cov_c*(vector-mean_FG)'+ 64*log(2*pi)+log(det(cov_c))-2*log(prior_cheetah);
        prob_grass = (vector - mean_BG)*inv_cov_g*(vector-mean_BG)'+ 64*log(2*pi)+log(det(cov_g))-2*log(prior_grass);
        
        % Assign each block to the class with highest posterior probability
        if prob_cheetah < prob_grass
            A(i,j) = 1;
        else
            A(i,j) = 0;
        end
        %Calculate Total Error
        error = error + abs(mask(i,j)-A(i,j)); 
    end
end

%Calculate Error Percentage
error_pct  = error/ 68850;
% Display the resulting image
imagesc(A);
colormap(gray(255))

simg = imread('cheetah.bmp');
dimg = im2double(simg);
mask = im2double(imread('cheetah_mask.bmp'));
error = 0;

% Pad the image with 4 rows of zeros
img = padarray(dimg, [0 0], 0, 'post');

% Define the traversal order
order = [0   1   5   6  14  15  27  28;
         2   4   7  13  16  26  29  42;
         3   8  12  17  25  30  41  43;
         9  11  18  24  31  40  44  53;
       	10  19  23  32  39  45  52  54;
       	20  22  33  38  46  51  55  60;
       	21  34  37  47  50  56  59  61;
       	35  36  48  49  57  58  62  63];

% Initialize the vector
vector = zeros(1, 8);

cov_c = cov(TrainsampleDCT_FG(:,1:8));
cov_g = cov(TrainsampleDCT_BG(:,1:8));
inv_cov_c = inv(cov_c);
inv_cov_g = inv(cov_g);
mean_FG_8 = mean_FG(1:8);
mean_BG_8 = mean_BG(1:8);

% Compute the state variable Y for each block in the image
A = zeros((size(img,1)), (size(img,2)));

for i = 4:size(img,1)-4
    for j = 4:size(img,2)-4
        % Extract an 8x8 block from the image
        block = img((i-4)+1:i+4, (j-4)+1:j+4);
        
        B = dct2(block);
        
        % Traverse the matrix in the given order and fill the vector
        for k = 0:7
            [row, col] = find(order==k);
            vector(k+1) = B(row, col);
        end

        % Compute the feature X (index of DCT coefficient with second-largest energy)
       
        prob_cheetah = (vector - mean_FG_8)*inv_cov_c*(vector-mean_FG_8)'+ 8*log(2*pi)+log(det(cov_c))-2*log(prior_cheetah);
        prob_grass = (vector - mean_BG_8)*inv_cov_g*(vector-mean_BG_8)'+ 8*log(2*pi)+log(det(cov_g))-2*log(prior_grass);
        
        % Assign each block to the class with highest posterior probability
        if prob_cheetah < prob_grass
            A(i,j) = 1;
        else
            A(i,j) = 0;
        end
        %Calculate Total Error
        error = error + abs(mask(i,j)-A(i,j)); 
    end
end

%Calculate Error Percentage
error_pct  = error/ 68850;
% Display the resulting image
imagesc(A);
colormap(gray(255))

order = [0   1   5   6  14  15  27  28;
         2   4   7  13  16  26  29  42;
         3   8  12  17  25  30  41  43;
         9  11  18  24  31  40  44  53;
       	10  19  23  32  39  45  52  54;
       	20  22  33  38  46  51  55  60;
       	21  34  37  47  50  56  59  61;
       	35  36  48  49  57  58  62  63];

% Initialize the vector
vector_8 = zeros(1, 8);

% Compute the state variable Y for each block in the image
A_8 = zeros((size(img,1)), (size(img,2)));
TrainsampleDCT_FG_8 = zeros(250,8);
TrainsampleDCT_BG_8 = zeros(1053,8);

for j = 1:8 
    TrainsampleDCT_FG_8(:,j) = TrainsampleDCT_FG(:,best(j));
    TrainsampleDCT_BG_8(:,j) = TrainsampleDCT_BG(:,best(j));
    mean_BG_8(j) = mean_BG(best(j));
    mean_FG_8(j) = mean_FG(best(j));
end

cov_c_8 = cov(TrainsampleDCT_FG_8);
cov_g_8 = cov(TrainsampleDCT_BG_8);
inv_cov_c_8 = inv(cov_c_8);
inv_cov_g_8 = inv(cov_g_8);

for i = 4:size(img,1)-4
    for j = 4:size(img,2)-4
        % Extract an 8x8 block from the image
        block = img((i-4)+1:i+4, (j-4)+1:j+4);
        
        B = dct2(block);
        
        % Traverse the matrix in the given order and fill the vector
        for k = 0:numel(B)-1
            [row, col] = find(order==k);
            vector(k+1) = B(row, col);
        end

        for j = 1:8 
            vector_8(j) = vector(best(j));
        end

        % Compute the feature X (index of DCT coefficient with second-largest energy)
       
        prob_cheetah = (vector_8 - mean_FG_8)*inv_cov_c_8*(vector_8-mean_FG_8)'+ 8*log(2*pi)+log(det(cov_c_8))-2*log(prior_cheetah);
        prob_grass = (vector_8 - mean_BG_8)*inv_cov_g_8*(vector_8-mean_BG_8)'+ 8*log(2*pi)+log(det(cov_g_8))-2*log(prior_grass);
        
        % Assign each block to the class with highest posterior probability
        if prob_cheetah < prob_grass
            A_8(i,j) = 1;
        else
            A_8(i,j) = 0;
        end
        %Calculate Total Error
        error = error + abs(mask(i,j)-A_8(i,j)); 
    end
end

%Calculate Error Percentage
error_pct  = error/ 68850;
% Display the resulting image
imagesc(A_8);
colormap(gray(255))
