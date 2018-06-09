/******************************************************************//**
* @brief   Generic interface for rendering text.
*
* @author  gernot
* @date    2018-02-06
* @version 1.0
**********************************************************************/
#pragma once
#ifndef IFont_h_INCLUDED
#define IFont_h_INCLUDED


// includes

#include <IDrawType.h>


/******************************************************************//**
* @brief   Namespace for renderer.
*
* @author  gernot
* @date    2018-03-17
* @version 1.0
**********************************************************************/
namespace Render
{


class ITextureLoader;
class IDrawBufferPrivider;


//---------------------------------------------------------------------
// IFont
//---------------------------------------------------------------------


/******************************************************************//**
* @brief   Generic interface for rendering text with a font.
*
* @author  gernot
* @date    2018-03-17
* @version 1.0
**********************************************************************/
class IFont
{
public: 

  virtual ~IFont() = default;

  virtual void Destroy( void ) = 0;                        //!< destroy all internal objects and cleanup
  virtual bool Load( Render::ITextureLoader &loader ) = 0; //!< load the glyphs

  //! calculates box of a string in relation to its height (maximum height of the font from the bottom to the top)
  virtual bool CalculateTextSize( const char *str, float height, float &box_x, float &box_btm, float &box_top ) = 0;

  //! render a texture based text
  virtual bool DrawText( IDrawBufferPrivider &buffer_provider, size_t textur_binding_id, const char *str, float height, float width_scale, const Render::TPoint3 &pos ) = 0; 
};


} // Render

#endif // IFont_h_INCLUDED
