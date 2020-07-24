use error_chain::*;

error_chain!{
    foreign_links {
        Image(image::error::ImageError);
        ObjectLoader(tobj::LoadError);
    }
}