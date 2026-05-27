# Differential Gaussian Rasterization

This is the modified version of differential Gaussian rasterization. 

## Usage
There is the sample code
```python
bg_image_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_texture/{viewpoint_cam.image_name}.png"
img = Image.open(bg_image_path).convert("RGB")
bg = transform(img).to(torch.float32).cuda()

bg_depth_pt_path = f"/mnt/data1/syjintw/NEU/dataset/hotdog/mesh_depth/{viewpoint_cam.image_name}.pt"
bg_depth = torch.load(background_depth_pt_path).unsqueeze(0).to("cuda")

render_pkg = render(viewpoint_cam, gaussians, pipe, bg, bg_depth)
```

```python
raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        depth=bg_depth # [YC]
        # antialiasing=pipe.antialiasing
    )
```

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>